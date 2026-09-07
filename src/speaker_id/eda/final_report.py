"""Synthesize completed computational EDA with explicit auditory/identity limits."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path


ESSENTIAL_SUMMARIES = (
    "signal_summary.json", "duplicate_summary.json", "split_summary.json",
    "archive_anomalies_summary.json", "forensics_summary.json", "embedding_summary.json",
    "embedding_geometry_summary.json", "semantic_summary.json", "calibration_summary.json",
    "metric_contract.json", "model_assets.json", "model_runtime_inventory.json",
)
ESSENTIAL_CSV = (
    "forensics.csv", "forensics_near_empty.csv", "embedding_files.csv", "embedding_segments.csv",
    "embedding_gain.csv", "embedding_neighbors.csv", "embedding_speakers.csv",
    "embedding_candidate_pairs.csv", "embedding_projection.csv", "fold_feature_comparison.csv",
    "semantic_files.csv", "semantic_segments.csv", "semantic_language.csv",
)
FIGURES = {
    "forensic_quantization.png": ("مقادیر واقعاً ذخیره‌شده در PCM", "دامنهٔ کد و آنتروپی توصیف نمونه‌های ذخیره‌شده‌اند؛ کیفیت ADC، فهم‌پذیری گفتار یا امکان بازیابی را اندازه نمی‌گیرند."),
    "forensic_target_spectrogram.png": ("ساختار طیفی نمونهٔ بسیار ضعیف", "شباهت طیفی یا توان پهن‌باند، علت خرابی یا وجود گفتار قابل استفاده را اثبات نمی‌کند؛ شکل جای شنیدن را نمی‌گیرد."),
    "embedding_geometry.png": ("هندسهٔ embedding ثابت", "مقایسهٔ شباهت برچسب‌ها و PCA روی کل داده فقط توصیفی است. نقاط نمودار، خوشه‌های هویت احرازشده نیستند؛ PCA این شکل نباید وارد preprocessing ارزیابی شود."),
    "embedding_stability.png": ("سطح سیگنال و ثبات درون فایل", "اختلاف embedding بین قطعه‌ها ممکن است از سکوت، نویز، محتوا یا کانال ناشی شود؛ تشخیص قطعی چندگویندگی یا تغییر جلسه نیست."),
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def esc(value) -> str:
    return html.escape(str(value))


def number(value, digits=0) -> str:
    if value is None:
        return "ثبت نشده"
    return f"{value:,.{digits}f}".translate(str.maketrans("0123456789,", "۰۱۲۳۴۵۶۷۸۹٬"))


def percentage(value, digits=1) -> str:
    return number(100 * value, digits) + "٪" if value is not None else "ثبت نشده"


def table(headers, rows) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers) + '</tr></thead><tbody>' + "".join('<tr>' + "".join(f"<td>{esc(cell)}</td>" for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def build_final_report(root: Path, output: Path | None = None) -> dict:
    root = root.resolve()
    output = (output or root / "reports/eda").resolve()
    processed = root / "data/processed/eda_v1"
    sources = [output / name for name in ESSENTIAL_SUMMARIES + ESSENTIAL_CSV]
    sources += [output / "figures" / name for name in FIGURES]
    sources += [processed / name for name in ("audio_manifest.csv", "folds.csv", "label_map.json", "calibration_roles.csv")]
    sources.append(output / "index.html")
    sources.append(root / "docs/model_audit_runtime.fa.md")
    missing = [str(path.relative_to(root)) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Final EDA report requires completed artifacts: " + ", ".join(missing))
    summaries = {name: json.loads((output / name).read_text(encoding="utf-8")) for name in ESSENTIAL_SUMMARIES}
    manifest = read_csv(processed / "audio_manifest.csv")
    manifest_hash = digest(processed / "audio_manifest.csv")
    names = {row["audio_file"] for row in manifest}
    if len(manifest) != 4529 or len(names) != len(manifest):
        raise ValueError("Final report requires all 4,529 unique original files")
    for name in ("duplicate_summary.json", "forensics_summary.json", "embedding_summary.json",
                 "embedding_geometry_summary.json", "semantic_summary.json", "archive_anomalies_summary.json"):
        summary = summaries[name]
        if summary.get("manifest_sha256", summary.get("input_manifest_sha256")) != manifest_hash:
            raise ValueError(f"Manifest mismatch in {name}")
    embedding, geometry, semantic = (summaries[name] for name in
                                    ("embedding_summary.json", "embedding_geometry_summary.json", "semantic_summary.json"))
    if embedding["status"] != "complete" or embedding["source_files"] != len(manifest):
        raise ValueError("Complete all-file embedding audit is required")
    if geometry["status"] != "descriptive_audit_complete" or semantic["status"] != "completed":
        raise ValueError("Complete geometry and semantic screening are required")
    if geometry["embedding_summary_sha256"] != digest(output / "embedding_summary.json"):
        raise ValueError("Geometry was built from a different embedding summary")
    embedded_rows = read_csv(output / "embedding_files.csv")
    if len(embedded_rows) != len(manifest) or {r["audio_file"] for r in embedded_rows} != names:
        raise ValueError("Embedding audit rows must retain every original filename")
    folds = read_csv(processed / "folds.csv")
    if len(folds) != len(manifest) or {r["audio_file"] for r in folds} != names:
        raise ValueError("Outer folds must retain every original filename")
    for key in ("calibration_summary.json", "metric_contract.json"):
        for filename, expected in summaries[key]["source_sha256"].items():
            if digest(processed / filename) != expected:
                raise ValueError(f"Outdated input {filename} in {key}")
    for summary, field in ((embedding, "artifact_sha256"), (geometry, "artifact_sha256"), (semantic, "output_sha256")):
        for filename, expected in summary[field].items():
            if digest(output / filename) != expected:
                raise ValueError(f"Outdated detailed artifact {filename}")
    optional = {}
    for pattern in ("quality_review*.json", "*waveform*summary.json", "*candidate*summary.json", "*verification*summary.json", "*overlap*summary.json"):
        for path in sorted(output.glob(pattern)):
            if path.name not in summaries and path.name not in optional:
                optional[path.name] = json.loads(path.read_text(encoding="utf-8"))
                sources.append(path)
    for pattern in ("quality_review*.csv", "*waveform*.csv", "*verification*.csv", "*overlap*.csv"):
        sources.extend(sorted(output.glob(pattern)))
    semantic_figure = output / "figures/semantic_screening.png"
    if semantic_figure.exists():
        sources.append(semantic_figure)
    builder = Path(__file__).resolve()
    entrypoint = root / "scripts/eda/build_final_report.py"
    sources.extend((builder, entrypoint))
    input_hashes = {path.relative_to(root).as_posix(): digest(path) for path in sorted(set(sources))}
    zero = [row for row in manifest if row["has_nonzero_signal"] == "False"]
    hours = sum(float(row["duration_seconds"]) for row in manifest) / 3600
    forensic = summaries["forensics_summary.json"]
    near_empty = forensic["near_empty_global_coverage"]["matching_global_files"]
    target = forensic["groups"]["target_44ea12a5"]
    target_embedding = next(row for row in read_csv(output / "embedding_speakers.csv")
                            if row["speaker_id"] == "44ea12a5-af34-418e-a9c2-fa6b93b66fce")
    coverage, strata = semantic["coverage"], semantic["strata"]
    calibration = summaries["calibration_summary.json"]
    metrics = summaries["metric_contract.json"]
    assets = summaries["model_assets.json"]
    runtime_inventory = summaries["model_runtime_inventory.json"]
    duplicate = summaries["duplicate_summary.json"]
    archive = summaries["archive_anomalies_summary.json"]
    overlap = optional.get("embedding_overlap_summary.json")
    quality = optional.get("quality_review_summary.json")
    sampled_fraction = coverage["sampled_unique_original_seconds"] / coverage["dataset_seconds"]
    now = datetime.now(timezone.utc).isoformat()
    sections = []
    sections.append(f'''<header><p class="eyebrow">IAAA 2026 · Speaker Identification</p><h1>جمع‌بندی نهایی EDA محاسباتی</h1><p class="lead">داده، کیفیت سیگنال، embedding ثابت و قرارداد اعتبارسنجی بررسی شده‌اند؛ محدودیت‌های شنیداری و هویت همچنان صریح‌اند.</p><p><a href="index.html">گزارش مرحلهٔ اول و نمودارهای سراسری ←</a></p></header>
<nav aria-label="بخش‌های گزارش"><a href="#coverage">پوشش و تصمیم‌ها</a><a href="#forensic">سیگنال‌های مشکوک</a><a href="#geometry">embedding</a><a href="#semantic">غربالگری صوت</a><a href="#validation">اعتبارسنجی</a><a href="#evidence">شواهد</a></nav>
<main><div class="metrics"><article><strong>{number(len(manifest))}</strong><span>فایل اصلی؛ همه در ارزیابی حفظ شدند</span></article><article><strong>{number(hours, 2)}</strong><span>ساعت صوت decodeشده</span></article><article><strong>{number(embedding['embedded_files'])}</strong><span>فایل دارای embedding ثابت</span></article><article><strong>{number(len(zero))} + {number(near_empty)}</strong><span>کاملاً صفر + غیرصفر با حداکثر ۶۴ نمونهٔ غیرصفر</span></article></div>
<aside class="notice"><strong>مرز نتیجه:</strong> هیچ بررسی شنیداری انسانی ثبت نشده است. مدل‌های عمومی فقط غربالگری و توصیف انجام داده‌اند؛ هویت افراد unknown، استقلال جلسه‌های ضبط و تک‌گوینده بودن تمام فایل‌ها احراز نشده‌اند. هیچ نتیجهٔ آموزش، Macro-F1 مدل یا کالیبراسیون اجراشده در این صفحه گزارش نمی‌شود.</aside>''')
    status_rows = [
        ("سلامت و ساختار", "تمام ۴۵۲۹ فایل تا پایان خوانده شدند؛ WAVهای دارای پسوند MP3 شناسایی شدند", "سلامت container تضمین وجود گفتار نیست", "decoder از محتوا؛ دادهٔ خام ثابت"),
        ("سیگنال و VAD", "آمار تمام نمونه‌ها و دو حالت VAD؛ آزمون حساسیت سطح", "VAD خام و تقویت دامنه، برچسب گفتار نیستند", "مسیر بدون حذف VAD برای خط مبنا محفوظ"),
        ("ممیزی PCM", f"{number(forensic['successful_files'])} فایل؛ {number(near_empty)} مورد تقریباً خالی در پوشش سراسری تأیید شد", "علت خرابی و بازیابی‌پذیری مشخص نشده", "بازبینی کیفیت جدا از ماسک آموزش"),
        ("embedding ثابت", f"{number(embedding['embedded_files'])} فایل غیرصفر و {number(embedding['segments'])} قطعه", "حداکثر ۱۸ ثانیه در هر فایل؛ شباهت، اثبات هویت یا جلسه نیست", "نامزدهای ناسازگاری برای بررسی؛ بدون اصلاح خودکار برچسب"),
        ("غربالگری محتوایی", f"{number(coverage['selected_files'])} فایل منتخب؛ {percentage(sampled_fraction, 2)} از مدت کل نمونه‌برداری شد", "نمونه‌گیری غنی از ناهنجاری و خروجی مدل؛ بدون شنیدن انسانی", "استنتاج شیوع زبان، موسیقی یا چندگویندگی ممنوع"),
        ("تقسیم و کالیبراسیون", "دو outer fold؛ نقش مستقل query و enrollment برای هر fold", "هویت unknown و جلسهٔ ضبط نامعلوم؛ enrollment کم‌نمونه", "تمام ۴۴۷ کلاس و تمام فایل‌ها در معیار اصلی؛ query از fit جدا"),
    ]
    sections.append('<section id="coverage"><h2>چه چیزهایی بررسی و چه تصمیم‌هایی تثبیت شدند</h2>' + table(("بخش", "انجام‌شده", "محدودیت", "تصمیم"), status_rows) + f'<p>تطابق CRC و SHA256 هر {number(archive["selected_files"])} فایل مشکوکِ انتخاب‌شده با ZIP اصلی تأیید شد. این ناهنجاری‌ها در نسخهٔ تحویلی وجود دارند؛ آزمون هدفمند، تأیید سلامت تمام آرشیو نیست.</p></section>')
    near_rows = [(row["audio_file"], "unknown" if row["speaker_id"] == "unknown" else "known", number(row["duration_seconds"], 3), number(row["nonzero_samples"]), number(row["unique_pcm_codes"])) for row in forensic["near_empty_files"]]
    sections.append(f'''<section id="forensic"><h2>کیفیت: غیرصفر بودن شرط کافی نیست</h2><p>{number(forensic['selection']['anomaly_files'])} فایل ناهنجار و {number(forensic['selection']['control_files'])} کنترل الگوریتمی روی تمام PCM آن‌ها بررسی شدند. کنترل‌ها گفتار تأییدشده نیستند و فقط {number(forensic['selection']['control_to_target_duration_ratio']['within_twenty_percent_files'])} مورد در فاصلهٔ ۲۰٪ مدت هدف قرار دارند؛ این مقایسه به‌عنوان مطالعهٔ کنترل‌شدهٔ همسان تفسیر نمی‌شود.</p>
<p>کلاس <code>44ea12a5-af34-418e-a9c2-fa6b93b66fce</code> پنج فایل با مدت کل {number(target['seconds'], 2)} ثانیه دارد. هر فایل فقط {number(target['metrics']['unique_pcm_codes']['min'])} تا {number(target['metrics']['unique_pcm_codes']['max'])} مقدار PCM متمایز دارد و حدود {percentage(target['metrics']['zero_fraction']['median'], 2)} نمونه‌ها صفرند. تقویت دامنه نمونه‌هایی را که قبلاً صفر شده‌اند بازسازی نمی‌کند.</p>
<p>هشت فایل زیر در کل مجموعه حداکثر ۶۴ نمونهٔ غیرصفر دارند؛ شمارش آن‌ها با PCM دقیق تطبیق داده شده است. فایل دارای فقط یک یا دو نمونهٔ غیرصفر، موج پیوستهٔ گفتار در خود ندارد؛ بااین‌حال آستانهٔ ۶۴ یک پرچم بازبینی برای مقایسهٔ سیاست آموزش است. این هشت فایل از foldها یا نقش‌های فعلی حذف نشده‌اند؛ ماسک فنی فعلی همچنان فقط ۸۹ فایل صفر را کنار می‌گذارد.</p>''' + table(("فایل", "گروه برچسب", "مدت، ثانیه", "نمونهٔ غیرصفر", "کد متمایز"), near_rows) + '</section>')
    agreement = geometry["nearest_known_agreement_all_known"]
    healthy = geometry["nearest_known_agreement_healthy_subset"]
    sections.append(f'''<section id="geometry"><h2>embedding گوینده و نامزدهای ناسازگاری</h2><p>encoder عمومی <a href="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb">SpeechBrain ECAPA-TDNN</a> با وزن ثابت و revision ثبت‌شده استفاده شد. هر فایل حداکثر سه پنجرهٔ شش‌ثانیه‌ای از کانال اصلی با RMS بیشتر دارد؛ در نمای اصلی حذف VAD و افزایش دامنه اعمال نشد. مجموع صوت نمونه‌برداری‌شده {number(embedding['probe_seconds'] / 3600, 2)} ساعت است؛ تمام بخش‌های فایل بررسی محتوایی نشده‌اند.</p>
<p>پس از کنار گذاشتن همسایه‌های داخل همان گروه محتوای دقیق، نزدیک‌ترین known در {number(agreement['agree'])} از {number(agreement['count'])} فایل known با برچسب خود فایل موافق است ({percentage(agreement['fraction'], 2)}). در زیرمجموعهٔ دارای RMS حداقل −۵۰، مدت حداقل پنج ثانیه و کمتر از ۹۹٪ نمونهٔ صفر، این نسبت {percentage(healthy['fraction'], 2)} از {number(healthy['count'])} فایل است.</p>
<aside class="notice">این نسبت‌ها توافق همسایگی روی کل داده‌اند؛ accuracy اعتبارسنجی، Macro-F1 یا برآورد عملکرد لیدربرد نیستند. برچسب‌ها فقط برای ممیزی استفاده شده‌اند.</aside>
<p><strong>شاهد مشخص محدودیت مدل:</strong> همان کلاس بسیار ضعیف <code>44ea12a5-af34-418e-a9c2-fa6b93b66fce</code> با حدود ۹۹٫۳٪ نمونهٔ صفر، در {number(round(int(target_embedding['files']) * float(target_embedding['nearest_known_agreement_fraction'])))} از {number(int(target_embedding['files']))} فایل توافق همسایهٔ هم‌برچسب دارد و میانگین cosine هم‌برچسب آن {number(float(target_embedding['mean_same_label_cosine']), 6)} است. این سازگاری بالا همراه با پشتیبانی موجِ پراکنده و ضربه‌مانند دیده می‌شود؛ سازگاری مدل به‌تنهایی وجود اطلاعات مفید گوینده یا گفتار فهم‌پذیر را ثابت نمی‌کند.</p>
<p>{number(geometry['known_classes_with_any_neighbor_disagreement'])} کلاس حداقل یک ناسازگاری همسایه و {number(geometry['known_classes_with_no_neighbor_agreement'])} کلاس هیچ توافق همسایه‌ای ندارند. شباهت نزدیک، دلیل کافی برای ادغام فایل‌ها، تغییر UUID یا اعلام نشت نیست. سیگنال بسیار ضعیف یا تقریباً خالی می‌تواند embeddingهای مشابه ولی فاقد اطلاعات گوینده ایجاد کند؛ شباهت بالای unknown به known به‌تنهایی تعارض هویت نیست. پایین بودن ثبات درون فایل نیز چندگویندگی را ثابت نمی‌کند.</p>
<p><a href="embedding_candidate_pairs.csv">نامزدهای شباهت بین برچسب‌ها</a> · <a href="embedding_speakers.csv">خلاصهٔ هر گوینده</a> · <a href="fold_feature_comparison.csv">مقایسهٔ ویژگی‌های دو fold</a></p>
<p>در ممیزی اولیهٔ hash و waveform، {number(duplicate['signal_exact_file_groups'])} گروه تکراریِ دارای سیگنال وجود داشت و هیچ هم‌پوشانی غیرتکراری تازه‌ای تأیید نشد. نتیجهٔ بررسی‌های تکمیلی نامزدها، در صورت تولید، در بخش شواهد جدا ثبت می‌شود. unknown یک برچسب مشترک است؛ شمارش زوج‌های نزدیک یا خوشه‌ها تعداد افراد واقعی را احراز نمی‌کند.</p></section>''')
    if overlap:
        selected = overlap["selection"]
        sections.append(f'''<section><h2>پیگیری نامزدهای embedding با waveform</h2><p>در {number(selected['selected_pairs'])} زوج منتخب با شرایط کیفیت مشخص، waveform کامل فایل‌ها بررسی شد؛ {number(overlap['successful_pairs'])} بررسی موفق و {number(overlap['verified_pairs'])} زوج مطابق معیار عملیاتی هم‌پوشانی تأیید شد. دو موج با شباهت embedding زیاد الزاماً صوت مشترک ندارند؛ این پیگیری نیز جست‌وجوی جامع تمام زوج‌ها نیست.</p>
<p>هر {number(selected['extreme_candidate_quality']['top20']['pairs'])} زوج ابتدای فهرست شباهت، دست‌کم یک فایل با RMS کمتر از −۵۰ dBFS داشت. بنابراین بالاترین cosineها در این داده به بررسی کیفیت نیاز دارند. پنجره‌های ثابت، هم‌پوشانی کمتر از چهار ثانیه و تغییر سرعت یا پردازش غیرخطی ممکن است از این روش پنهان بمانند. fold، برچسب و گروه محتوا بر اساس این نامزدها تغییر نکردند.</p><p><a href="embedding_overlap_summary.json">روش انتخاب، آستانهٔ تأیید و محدودیت‌های پیگیری</a></p></section>''')
    strata_rows = []
    for key, label in (("healthy_controls", "کنترل‌های الگوریتمی"), ("anomaly_and_listening_union", "ناهنجاری‌ها و صف نمونه‌ها")):
        row = strata[key]
        strata_rows.append((label, number(row["interpretable_original_segments"]), number(row["ast_speech_ge_05_segments"]), number(row["ast_music_ge_05_segments"]), number(row["ast_speech_and_music_ge_05_segments"])))
    sections.append(f'''<section id="semantic"><h2>غربالگری محتوایی خودکار، با پوشش مشخص</h2><p><a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">AST عمومی AudioSet</a> روی پنجره‌های منتخب و <a href="https://huggingface.co/openai/whisper-base">Whisper base عمومی</a> روی زیرمجموعه‌ای کوچک اجرا شدند. از {number(coverage['selected_files'])} فایل منتخب، {number(coverage['original_windows'])} پنجرهٔ اصلی با مجموع {number(coverage['sampled_unique_original_seconds'], 2)} ثانیه نمونه‌برداری شد؛ {number(coverage['ast_scored_original_windows'])} پنجره امتیاز AST گرفت و {number(coverage['inconclusive_signal_original_windows'])} پنجره از نظر سیگنال نامطمئن باقی ماند.</p>
<p>کنترل‌ها {number(coverage['healthy_control_files'])} فایل هستند. آستانهٔ ۰٫۵ فقط برای توصیف امتیازهای کالیبره‌نشدهٔ AST استفاده شد؛ دسته‌ها چندبرچسبی‌اند و ممکن است هم‌پوشانی داشته باشند.</p>''' + table(("زیرمجموعه", "پنجرهٔ قابل تفسیر مدل", "امتیاز گفتار ≥۰٫۵", "امتیاز موسیقی ≥۰٫۵", "هر دو"), strata_rows) + f'''<p>Whisper برای {number(coverage['whisper_scored_files'])} فایل از {number(coverage['whisper_selected_files'])} فایل منتخب، در مجموع {number(coverage['whisper_scored_seconds'])} ثانیه اجرا شد. خروجی زبان و متن، حقیقت تأییدشده و تخمین شیوع در کل داده نیست؛ برای پرهیز از برداشت نادرست، متن‌ها و برچسب‌های تفصیلی در صفحهٔ اصلی نمایش داده نمی‌شوند.</p>
<p>{number(coverage['gain_comparator_files'])} فایل نیز مقایسهٔ جداگانهٔ افزایش دامنه داشتند. پنجره‌های صفر، RMS کمتر از −۸۰ dBFS یا بیش از ۹۹٪ نمونهٔ صفر، حتی با خروجی مدل، نامطمئن باقی می‌مانند. هیچ مشاهدهٔ شنیداری انسانی، لهجه، ویژگی جمعیت‌شناختی یا چندگویندگی قطعی ثبت نشده است.</p></section>''')
    fold_rows = []
    for row in calibration["folds"]:
        counts = row["role_file_counts"]
        fold_rows.append((number(row["outer_fold"]), number(counts["outer_validation"]), number(counts["known_enrollment"]), number(counts["known_calibration_query"]), number(counts["unknown_calibration_query"]), number(row["encoder_fit_files"]), number(len(row["known_singleton_enrollment_only"])) ))
    sections.append('<section id="validation"><h2>قرارداد ارزیابی و کالیبراسیون، پیش از آموزش</h2><p>تمام ۴۵۲۹ فایل در دو outer fold حفظ شدند. هر دو fold، همهٔ ۴۴۶ کلاس known را در enrollment دارند. داخل بخش توسعه، یک گروه مستقل از هر کلاس با حداقل دو گروه به query اختصاص یافت؛ singletonها فقط enrollment هستند. unknown در سطح گروه محتوا تقسیم شد و استقلال هویت آن ادعا نمی‌شود.</p>' + table(("outer fold", "ارزیابی اصلی", "enrollment known", "query known", "query unknown", "مجاز برای fit", "کلاس singleton"), fold_rows) + f'''<p>فایل‌های query از آموزش encoder، prototype و تبدیل‌های یادگرفتنی کنار گذاشته شده‌اند؛ outer validation نیز وارد fit یا تنظیم آستانه نمی‌شود. این‌ها نقش‌های آماده‌شده‌اند و هیچ encoder یا calibrator روی دادهٔ مسابقه fit نشده است. پس از تغییر شمار enrollment یا آموزش نهایی، راهبرد کالیبراسیون باید دوباره مستند شود.</p>
<p>معیار اصلی میانگین F1 بر همهٔ ۴۴۷ برچسب ثابت، یک‌بار روی پیش‌بینی‌های تجمیع‌شدهٔ خارج از fold است. هیچ فایل صفر یا کلاس کم‌نمونه از نمره حذف نمی‌شود. میانگین نمرهٔ دو fold جای نمرهٔ تجمیع‌شده را نمی‌گیرد. نام فایل تکراری، مفقود، اضافه یا برچسب نامعتبر رد می‌شود.</p>
<details><summary>دو محاسبهٔ تحلیلی برای کنترل قرارداد معیار</summary><p>پیش‌بینی unknown برای همهٔ فایل‌ها: Macro-F1 برابر <code>{metrics['all_unknown_baseline']['macro_f1']:.10f}</code>. سناریوی فرضی «همهٔ غیرصفرها درست و همهٔ صفرها unknown»: <code>{metrics['perfect_nonzero_with_zero_to_unknown_diagnostic']['macro_f1']:.10f}</code>. دومی با استفاده از حقیقت برچسب محاسبه شده و نتیجهٔ مدل یا سقف عمومیِ قابل دستیابی نیست.</p></details>
<p><a href="../../data/processed/eda_v1/calibration_roles.csv">نقش همهٔ فایل‌ها در هر outer fold</a> · <a href="calibration_summary.json">قرارداد کالیبراسیون</a> · <a href="metric_contract.json">قرارداد معیار و محاسبات تحلیلی</a></p></section>''')
    sections.append('<section id="figures"><h2>شواهد تصویری مکمل</h2>')
    for name, (title, caption) in FIGURES.items():
        sections.append(f'<figure><h3>{esc(title)}</h3><img src="figures/{esc(name)}" alt="{esc(title)}" loading="lazy"><figcaption>{esc(caption)}</figcaption></figure>')
    if semantic_figure.exists():
        sections.append('<figure><h3>پوشش و امتیازهای غربالگری خودکار</h3><img src="figures/semantic_screening.png" alt="امتیازهای مدل در نمونه‌های منتخب" loading="lazy"><figcaption>نمونه‌ها تصادفی نیستند؛ فراوانی امتیازها، شیوع محتوای صوتی در کل داده را نشان نمی‌دهد.</figcaption></figure>')
    sections.append('</section><section id="evidence"><h2>شواهد، نسخه‌ها و کار باقی‌مانده</h2><p>گزارش محاسباتی به پایان رسیده است؛ شنیدن انسانیِ نمونه‌های مشکوک، احراز هویت unknown و جلسه‌های ضبط از این محاسبات نتیجه نمی‌شود. ناسازگاری‌های مدل باید به‌عنوان نامزد بررسی باقی بمانند. دادهٔ خام و برچسب‌ها بر اساس این صفحه خودکار اصلاح نمی‌شوند.</p>')
    if optional:
        sections.append('<p>خروجی‌های تکمیلی موجود هنگام ساخت گزارش:</p><ul>' + ''.join(f'<li><a href="{esc(name)}">{esc(name)}</a></li>' for name in optional) + '</ul>')
        if (output / "quality_review.csv").exists():
            sections.append('<p><a href="quality_review.csv">فهرست تلفیقی بازبینی کیفیت برای تمام فایل‌ها</a>؛ پرچم بازبینی را با تأیید خرابی، نبود گفتار یا اصلاح برچسب یکسان نگیرید.</p>')
    if quality:
        sections.append(f'''<p>در فهرست تلفیقی، {number(quality['flagged_files'])} فایل از {number(quality['source_files'])} فایل دست‌کم یک پرچم بازبینی دارند؛ {number(quality['unflagged_files'])} فایلِ بدون پرچم نیز «صوت پاکِ تأییدشده» نیستند. پرچم‌ها هم‌پوشانی دارند. برای {number(quality['additional_near_empty_exclusion_ablation_candidates'])} فایل تقریباً خالی، کنار گذاشتن از آموزش به‌عنوان مقایسهٔ از پیش مشخص‌شده پیشنهاد شده و روی foldهای ثابت اعمال نشده است.</p>''')
    sections.append('<h3>وزن‌های مستقل و تفاوت محیط پژوهش با لیدربرد</h3><p>فایل‌های همان سه مدل عمومی و ثابت، همراه تنظیمات و tokenizer موردنیاز، اکنون در <code>artifacts/models/</code> همین پروژه قرار دارند. تطابق بایت‌به‌بایت آن‌ها در registry ثبت شده است؛ این کپی، اجرای دوبارهٔ inference یا آزمون سازگاری در محیط جدید نیست. مدل‌ها فقط برای ممیزی پژوهشی استفاده شدند و checkpoint آموزش‌دیده روی مسابقه وارد نشده است.</p>')
    runtime_rows = [(package, embedding["versions"][package], semantic["runtime"][package], assets["known_leaderboard_range_conflicts"][package]["guide"])
                    for package in ("numpy", "scipy", "soundfile")]
    sections.append(table(("پکیج", "اجرای embedding", "اجرای غربالگری", "بازهٔ راهنمای لیدربرد"), runtime_rows))
    sections.append(f'<p>هر سه نسخهٔ پژوهشیِ این جدول بیرون از بازهٔ اعلام‌شدهٔ لیدربرد هستند. هیچ آزمون inference در محیط واقعی لیدربرد یا Linux مطابق آن انجام نشده است. انتخاب decoder، وابستگی‌های کمینه و اجرای بدون شبکه باید در مرحلهٔ بسته‌بندی روی محیط هدف آزموده شوند. فهرست {number(runtime_inventory["package_count"])} پکیج محیط پژوهش شامل پکیج‌های استفاده‌نشده هم هست؛ lock یا اثبات بازتولید نصب نیست.</p><p><a href="model_assets.json">registry وزن‌ها و SHA256</a> · <a href="model_runtime_inventory.json">فهرست محیط پژوهشی</a> · <a href="../../docs/model_audit_runtime.fa.md">روش انتقال وزن‌ها و بازسازی پیشنهادی محیط</a></p><p>این گزارش شامل آموزش مسابقه، ثبت MLflow یا نتیجهٔ استقرار نیست.</p>')
    artifact_links = [("ممیزی PCM", "forensics_summary.json"), ("استخراج embedding", "embedding_summary.json"), ("هندسهٔ embedding", "embedding_geometry_summary.json"), ("غربالگری خودکار", "semantic_summary.json"), ("اثر فایل‌های صفر بر معیار", "metric_contract.json"), ("داده و نمودارهای مرحلهٔ اول", "index.html")]
    sections.append('<div class="links">' + ''.join(f'<a href="{esc(path)}">{esc(label)}</a>' for label, path in artifact_links) + '</div>')
    sections.append('<details><summary>فهرست ورودی‌ها و SHA256 برای بازتولید</summary>' + table(("فایل نسبت به پروژه", "SHA256"), list(input_hashes.items())) + '</details>')
    sections.append(f'<footer>ساخت گزارش به وقت UTC: <code>{esc(now)}</code> · <a href="final_report_stats.json">وضعیت و hashهای ماشینی</a></footer></section></main>')
    css = '''*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:#f4f6fb;color:#172842;font:15px/1.95 Tahoma,Arial,sans-serif}header{padding:46px max(24px,calc((100vw - 1160px)/2));background:#142741;color:white}header a{color:#b7d6ff}h1{font-size:34px;line-height:1.65;margin:6px 0 12px}h2{font-size:24px;margin:0 0 20px}h3{font-size:18px}.eyebrow{direction:ltr;text-align:right;letter-spacing:.07em;font:12px Arial;color:#b1c9e7}.lead{max-width:930px;font-size:18px;color:#e4ecf8}nav{position:sticky;top:0;background:#fffef8f7;border-bottom:1px solid #dbe2ee;display:flex;gap:22px;justify-content:center;flex-wrap:wrap;padding:12px 20px;z-index:3}a{color:#20549c;text-decoration:none}a:hover{text-decoration:underline}main{max-width:1208px;margin:auto;padding:28px 24px 55px}section{background:white;border:1px solid #dde5ef;border-radius:16px;padding:30px;margin-top:28px}p{margin:13px 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}.metrics article{background:white;padding:22px 18px;border-radius:14px;border:1px solid #dde5ef}.metrics strong{font-size:30px;display:block;color:#173f72;line-height:1.6}.metrics span{font-size:13px;color:#556981}.notice{padding:18px 22px;border-right:5px solid #b97a1a;background:#fff6e5;border-radius:8px;margin:20px 0}.table-wrap{width:100%;overflow-x:auto;border:1px solid #e0e7f0;border-radius:10px}table{border-collapse:collapse;min-width:680px;width:100%;font-size:13px}th{background:#edf3fa;color:#264361;text-align:right;padding:12px;vertical-align:top}td{padding:12px;border-top:1px solid #e5ebf3;vertical-align:top;overflow-wrap:anywhere}tr:nth-child(even){background:#fafcff}code{direction:ltr;unicode-bidi:embed;display:inline-block;background:#eef3f9;padding:1px 5px;border-radius:4px;font:12px/1.7 Consolas,monospace;overflow-wrap:anywhere;max-width:100%}figure{margin:26px 0 40px;border:1px solid #e3e9f1;border-radius:12px;padding:14px}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#56677d;padding:12px 8px}details{margin-top:22px;border:1px solid #dce5f1;border-radius:10px;padding:14px}summary{cursor:pointer;color:#234f86;font-weight:bold}.links{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}.links a{border:1px solid #d4e2f3;border-radius:8px;padding:7px 12px;background:#f3f7fc}footer{font-size:12px;color:#66788f;margin-top:24px}@media(max-width:800px){header{padding:28px 22px}h1{font-size:26px}main{padding:20px 12px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:22px 16px}nav{position:static;font-size:13px;gap:12px}h2{font-size:21px}}@media print{nav{position:static}body{background:white}section,figure{break-inside:avoid}details{display:none}a{color:inherit}}'''
    document = '<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>جمع‌بندی نهایی EDA محاسباتی</title><style>' + css + '</style></head><body>' + ''.join(sections) + '</body></html>'
    page = output / "final.html"
    page.write_text(document, encoding="utf-8")
    stats = {"report_version": "final_computational_eda_v1", "generated_at_utc": now,
             "status": "computational_eda_complete_with_explicit_unresolved_limits",
             "human_listening_completed": False, "unknown_identities_verified": False,
             "recording_sessions_verified": False, "model_performance_estimated": False,
             "source_files": len(manifest), "all_original_files_preserved": True,
             "embedded_files": embedding["embedded_files"], "fully_zero_files": len(zero),
             "near_empty_nonzero_files": near_empty, "semantic_selected_files": coverage["selected_files"],
             "semantic_unique_sampled_seconds_fraction": sampled_fraction,
             "manifest_sha256": manifest_hash, "input_sha256": input_hashes,
             "builder_sha256": digest(builder), "entrypoint_sha256": digest(entrypoint),
             "final_html_sha256": digest(page), "optional_summaries_included": sorted(optional),
             "figures_included": list(FIGURES) + ([semantic_figure.name] if semantic_figure.exists() else [])}
    (output / "final_report_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats
