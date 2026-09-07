# IAAA 2026 — Open-set speaker identification

این پروژه ۴۴۶ گویندهٔ شناخته‌شده و کلاس مشترک `unknown` را بررسی می‌کند. معیار مسابقه Macro-F1 روی ۴۴۷ کلاس است.

مرحلهٔ فعلی: EDA خودکار نهایی، شامل بررسی موج، مدل‌های عمومی ثابت، پرچم‌های بازبینی و قرارداد اعتبارسنجی/کالیبراسیون. [گزارش نهایی](reports/eda/final.html) و [جمع‌بندی تصمیم‌ها و محدودیت‌ها](docs/eda_final.fa.md) آماده‌اند. بازبینی شنیداری انسانی و احراز هویت/session انجام نشده‌اند. آموزش آغاز نشده است. زیرساخت مدل اصلی CAM++، tracking و استقرار آمادهٔ بررسی است؛ وضعیت واقعی و مانع فعلی ظرفیت Vast در [گزارش زیرساخت](docs/infrastructure_readiness.fa.md) ثبت شده است.

## محیط قابل بازتولید

Python 3.12 و `uv` لازم است. از ریشهٔ پروژه:

```powershell
uv --cache-dir artifacts/tooling/uv-cache sync --locked --group eda
```

نسخه‌ها در `pyproject.toml` و `uv.lock` ثبت شده‌اند. وابستگی‌های گروه `eda`، از جمله matplotlib و WebRTC VAD، ابزار پژوهش محلی هستند و قرارداد محیط inference لیدربرد را تعریف نمی‌کنند. Python/NumPy/SciPy/SoundFile انتخاب‌شده در محدودهٔ مرتبط اعلامی مسابقه قرار دارند؛ سازگاری سرور ارزیاب هنوز باید در محیط واقعی آن آزموده شود.

## اجرای مرحله‌ها

فرمان‌ها را به ترتیب از ریشهٔ پروژه اجرا کنید. `uv run` از محیط همین پروژه استفاده می‌کند.

```powershell
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/inspect_inventory.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/audit_audio.py --workers 4
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/audit_duplicates.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/audit_archive.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/audit_vad_sensitivity.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/build_splits.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/prepare_listening.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/eda/build_report.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/checks/run_tests.py
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python scripts/checks/check_eda.py
```

اسکن موج، فایل‌ها را به‌طور کامل می‌خواند، در چهار worker محدود اجرا می‌شود و checkpoint دارد. cache فقط در صورت تطابق امضای کد/پکیج‌ها و مشخصات فایل استفاده می‌شود؛ این سازوکار بر ثابت‌بودن دادهٔ خام تکیه دارد. برای خواندن دوبارهٔ تمام بایت‌ها `--no-resume` بدهید. اسکن‌های `--limit` را فقط با مسیر خروجی جدا اجرا کنید تا گزارش اصلی با آزمایش کوچک جایگزین نشود.

## خروجی‌ها

خروجی اصلی فعلی `reports/eda/final.html` است. نسخهٔ اول و گزارش موج برای سابقه حفظ شده‌اند. وزن‌های عمومی مورد استفاده با hash کنترل‌شده در `artifacts/models` همین پروژه قرار دارند؛ [محیط واقعی ممیزی مدل‌ها و روش بازسازی](docs/model_audit_runtime.fa.md) جدا از محیط لیدربرد مستند شده است.

یافته‌ها و تصمیم‌های فعلی در [نتیجهٔ EDA نسخهٔ ۱](docs/eda_decisions.fa.md) و پرسش‌های مشخص دربارهٔ کیفیت داده در [متن آمادهٔ پیگیری](docs/data_quality_questions.fa.md) ثبت شده‌اند؛ هیچ پیامی برای برگزارکننده ارسال نشده است.

- `reports/eda/index.html`: گزارش فارسی محلی و نمودارها؛ به اینترنت نیاز ندارد.
- `data/processed/eda_v1/audio_manifest.csv`: یک ردیف برای هر فایل، SHA256 فایل/PCM، فرمت و کیفیت موج و خروجی تشخیص گفتار.
- `reports/eda/signal_summary.json`: پوشش، تعریف واحدها و آستانه‌ها، نسخهٔ ابزارها و اثر انگشت داده.
- `reports/eda/duplicate_pairs.csv` و `duplicate_groups.csv`: شواهد تکرار/هم‌پوشانی و وضعیت تأیید آن‌ها؛ نبود تطابق، استقلال جلسهٔ ضبط را اثبات نمی‌کند.
- `data/processed/eda_v1/folds.csv`: در صورت امکان حفظ پوشش همهٔ کلاس‌ها، fold هر فایل و گروه محتوا را ثبت می‌کند؛ تقسیم موقت است. اگر دادهٔ کافی برای یک کلاس وجود نداشته باشد، فایل ساخته نمی‌شود و علت در خلاصه ثبت می‌شود.
- `data/processed/eda_v1/label_map.json`: ترتیب ثابت برچسب‌ها، با `unknown` در اندیس صفر.
- `reports/eda/split_summary.json`: پوشش کلاس‌ها و محدودیت‌های split.
- `reports/eda/split_support.csv`: تعداد گروه محتوای قابل استفاده برای هر گوینده و امکان اعتبارسنجی آن.
- `reports/eda/audio_samples/index.html`: نمونه‌های کوتاه برای بررسی شنیداری؛ آماده‌سازی این صفحه به معنی انجام بررسی شنیداری نیست.

برای مشاهدهٔ مستقیم، `reports/eda/index.html` را در مرورگر باز کنید. جهت دسترسی HTTP محلی به لینک‌های manifest و داده نیز می‌توان **فقط روی loopback** سرور راه انداخت:

```powershell
uv --cache-dir artifacts/tooling/uv-cache run --locked --group eda python -m http.server 8765 --bind 127.0.0.1 --directory .
```

سپس مسیر `http://127.0.0.1:8765/reports/eda/index.html` را باز کنید. سرویس صرفاً برای مشاهدهٔ محلی است.

## مرزهای تفسیر

`usable_for_training` در manifest فقط عبور از کنترل سلامت فایل و وجود نمونهٔ غیرصفر را نشان می‌دهد؛ تأیید گفتار قابل استفاده یا صحت برچسب نیست. clipping و انرژی پایین در این مرحله شاخص‌های بررسی‌اند. پیش‌بینی‌های WebRTC VAD حقیقت قطعی نیستند و هیچ نمونه‌ای به‌دلیل VAD به‌تنهایی حذف نمی‌شود.

`train_eligible` در folds علاوه بر سلامت سیگنال، تعارض برچسب در گروه محتوای تأییدشده را در نظر می‌گیرد. هنگام ارزیابی fold `k`، همهٔ ردیف‌های همان fold باید امتیاز بگیرند؛ آموزش فقط ردیف‌های `fold != k` و `train_eligible == True` را مصرف می‌کند. تنظیم threshold یا هر تبدیل قابل یادگیری باید داخل دادهٔ توسعه/آموزش همان fold انجام شود. فایل خام و برچسب‌ها تغییر نمی‌کنند.

هویت فردی اعضای `unknown` و استقلال جلسه‌های ضبط از دادهٔ فعلی اثبات نشده‌اند. ساختار embedding تمام غیرصفرها و بررسی خودکار نوع صدا/زبان روی نمونه‌های منتخب انجام شده است؛ بازبینی شنیداری انسانی انجام نشده است. نمرهٔ مدل، انسجام embedding و نبود پرچم به معنی تأیید گفتار یا برچسب نیستند.

## مرحلهٔ تکمیلی و کنترل نهایی

ممیزی مدل‌ها از وزن عمومی ثابت استفاده می‌کند و هیچ پارامتر مسابقه را آموزش نمی‌دهد. محیط پژوهشیِ استفاده‌شده نسخه‌هایی خارج از بازهٔ لیدربرد دارد؛ راهنمای محیط بالا را بخوانید. پس از اجرای `audit_embeddings.py` و `audit_semantic.py` طبق آن راهنما، بقیهٔ دستورها با محیط سبک پروژه اجرا می‌شوند:

```powershell
.venv\Scripts\python.exe scripts\eda\audit_forensics.py
.venv\Scripts\python.exe scripts\eda\audit_geometry.py
.venv\Scripts\python.exe scripts\eda\audit_embedding_overlap.py
.venv\Scripts\python.exe scripts\eda\build_calibration.py
.venv\Scripts\python.exe scripts\eda\build_quality_review.py
.venv\Scripts\python.exe scripts\eda\plot_semantic.py
.venv\Scripts\python.exe scripts\eda\build_final_report.py
.venv\Scripts\python.exe scripts\checks\run_tests.py
.venv\Scripts\python.exe scripts\checks\check_final_eda.py
.venv\Scripts\python.exe scripts\eda\stage_models.py --verify-only
```

`calibration_roles.csv` نقش query/enrollment/encoder-fit را برای هر outer fold ثبت می‌کند؛ queryها از آموزش encoder و مرجع‌سازی مستقل‌اند. `quality_review.csv` تمام فایل‌ها را با پرچم‌ها و پیشنهاد بررسی نگه می‌دارد. هشت فایل تقریباً خالی فعلاً فقط نامزد مقایسهٔ حذف‌اند؛ mask فعلی همچنان همان ۸۹ صفر را از آموزش کنار می‌گذارد. تمام فایل‌ها در ارزیابی باقی می‌مانند.

## نظم پروژه

منطق مشترک در `src/speaker_id`، ورودی‌های اجرا در `scripts`، آزمون‌ها در `tests` و تصمیم‌ها در `docs` نگهداری می‌شوند. دادهٔ اصلی، cache، محیط پایتون، کلیدها و خروجی‌های بزرگ در Git قرار نمی‌گیرند. هر مرحله خروجی مشخص و قابل بازتولید دارد؛ کد CAM++ و recipeهای B001/F001 اضافه شده‌اند؛ وزن عمومی در مسیر ignored محلی آماده است و هیچ وزنی روی دادهٔ مسابقه آموزش ندیده است.

## CAM++ و آماده‌سازی آموزش

[راهبرد آموزش](docs/training_strategy.fa.md) و [runbook زیرساخت و وضعیت واقعی](docs/infrastructure_readiness.fa.md) مرجع مرحلهٔ جاری‌اند. `configs/model/campp.json` مدل اصلی، `configs/train/campp_baseline.json` خط مبنای B001 و `configs/train/campp_finetune.json` گزینهٔ F001 را تعریف می‌کنند.

`python scripts/train.py` فقط قراردادها را بررسی می‌کند. شروع واقعی به پیام صریح کاربر، فلگ اجرا، readiness موفق خود instance و roundtrip زندهٔ MLflow وابسته است. کد از Git، داده/متادیتا/وزن از SSH و ZIP منتقل می‌شوند. گزارش‌های EDA و دادهٔ processed عمداً در مخزن عمومی نیستند؛ لینک‌های آن‌ها با نسخهٔ محلی کامل کار می‌کنند.
