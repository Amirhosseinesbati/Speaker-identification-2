# آماده‌سازی CAM++ و زیرساخت آموزش

تاریخ: ۲۰۲۶-۰۹-۰۷. **هیچ آموزش، optimizer step یا استخراج سراسری embedding در این مرحله اجرا نشده است.**

## وضعیت واقعی

- مدل اصلی CAM++ است. وزن عمومی VoxCeleb و معماری رسمی نسخه‌دار آماده‌اند؛ probe محلی روی فایل واقعی، embedding نرمال‌شدهٔ ۵۱۲بعدی و graph خروجی ۴۴۶کلاسه را تأیید کرده است. این probe روی CPU با runtime پژوهشی انجام شده و تأیید سازگاری CUDA/لیدربرد نیست.
- ۹۳ آزمون داده، نقش‌های کالیبراسیون، امتیازدهی، CAM++، انتقال ZIP، tracking و readiness گذشته‌اند. dry-run قرارداد ۴۵۲۹ فایل و دو fold را تأیید کرده است.
- Vast CLI رسمی نسخهٔ ۱٫۶٫۰ با اجازهٔ کاربر توسط uv نصب شد و توکن `.env` برای اتصال واقعی استفاده شد. درخواست شروع instance `50079023` پاسخ «Required resources are currently unavailable, state change queued.» داد؛ بررسی بعدی `actual_status=exited`, `intended_status=stopped`, `cur_state=stopped` را نشان داد.
- طبق تصمیم کاربر همان instance نگه داشته می‌شود؛ هیچ جایگزینی بررسی/اجاره و هیچ instance حذف نشده است.
- احراز هویت و خواندن MLflow از سیستم محلی موفق بود. نام experiment مستقل این مرحله `iaaa2026-campp-infrastructure-20260907` است. نتیجهٔ واقعی ایجاد run و roundtrip در `artifacts/infrastructure/mlflow_local_probe/preflight_result.json` ثبت می‌شود؛ نبود این فایل به معنی انجام‌نشدن آن مرحله است.
- SSH، استقرار واقعی، انتقال داده و token، نصب محیط GPU و probe از داخل 3090 **منتظر در دسترس‌شدن instance هستند**. گزارش نهایی `artifacts/infrastructure/readiness.json` باید تا آن زمان `blocked` باقی بماند.

## مدل و پروتکل

`configs/model/campp.json` معماری و frontend را قفل می‌کند؛ وابستگی inference به ModelScope، دریافت اینترنتی وزن یا TorchCodec ندارد. SoundFile بر اساس محتوای فایل decode می‌کند؛ frontend مشترک تک‌کاناله، 16 kHz، FBank هشتادبعدی و mean normalization است.

وزن عمومی:

```text
iic/speech_campplus_sv_en_voxceleb_16k @ v1.0.2
SHA256 5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de
3D-Speaker commit 065629c313eaf1a01c65c640c46d77e61e9607b4
```

`B001` مدل ثابت، prototype و آستانهٔ سراسری کالیبره‌شده را ارزیابی می‌کند. `F001` گزینهٔ بعدی fine-tuning انتهای CAM++ با AAM برای knownهاست. unknownها یک هویت مشترک در loss نیستند. inner query وارد encoder fit یا gallery نمی‌شود؛ outer برای انتخاب epoch/threshold مصرف نمی‌شود. سکوت‌ها در ارزیابی حفظ می‌شوند. شروع هرکدام نیازمند دستور صریح بعدی کاربر است.

## نسخه‌ها و چرخهٔ Git

کد و config فقط محلی نوشته می‌شوند، به شاخهٔ `develop` commit/push می‌شوند، سپس سرور فقط clone/pull و fast-forward می‌کند. هر مرحلهٔ provisioning، HEAD، origin، branch، hostname و instance فعلی را دوباره کنترل می‌کند. هیچ کد منبعی با SCP روی سرور ارسال نمی‌شود.

مخزن GitHub عمومی است. دادهٔ خام، متادیتای فایل‌ها و گزارش‌های ریز EDA در Git نیستند؛ نسخهٔ محلی آن‌ها حفظ شده است. متادیتای ضروری در ZIP جداگانه با چهار فایل allowlist منتقل می‌شود. `.env`، کلیدها، وزن‌ها و خروجی اجراها نیز از Git مستثنا هستند.

محیط train: Python 3.12، torch/torchaudio `2.10.0+cu128`، NumPy `2.2.6`، SciPy `1.15.3`، SoundFile `0.13.1`، MLflow client (`mlflow-skinny`) `3.7.0`، matplotlib `3.11.1`. کل dependency graph در `uv.lock` قفل شده است. `bootstrap_server.sh` با uv `0.11.28` همین محیط را نصب می‌کند. نسخه‌های core با بازه‌های راهنمای مسابقه مقایسه می‌شوند؛ فایل راهنما freeze دقیق سرور نیست و آزمون بستهٔ نهایی آفلاین همچنان مرحلهٔ تحویل خواهد بود.

## مراحل آمادهٔ اجرا پس از بازگشت ظرفیت سرور

از ریشهٔ workspace محلی، با کلید SSH ثبت‌شدهٔ حساب:

```powershell
uv --cache-dir artifacts/tooling/uv-cache sync --locked --group ops --group eda
.venv/Scripts/python.exe scripts/infra/vast_control.py show
# start فقط هنگام امکان بازگشت ظرفیت همان instance؛ هیچ حلقهٔ راه‌اندازی یا اجارهٔ جایگزین وجود ندارد.
.venv/Scripts/python.exe scripts/infra/vast_control.py start
.venv/Scripts/python.exe scripts/infra/vast_control.py show
.venv/Scripts/python.exe scripts/infra/prepare_transfer.py
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py inspect --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_ed25519
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py deploy --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_ed25519
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py bootstrap --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_ed25519
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py upload-assets --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_ed25519
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py verify --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_ed25519
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py download-evidence --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_ed25519
```

اولین SSH هنوز انجام نشده و تطبیق کلید حساب با کلید محلی پس از running بررسی می‌شود. در صورت خطای SSH، ابتدا `vast_control.py logs` خوانده می‌شود و علت تعیین می‌شود؛ کلید یا تنظیمات کورکورانه تغییر نمی‌کنند.

`upload-assets` فقط متادیتا، وزن عمومی، binding آزمایش جدید، ZIP خام و سه متغیر MLflow را انتقال می‌دهد. tokenهای Vast/Git روی سیستم محلی باقی می‌مانند. فایل `.env` سرور regular file با mode `600` است؛ انتقال رمزگذاری‌شده با SSH/SFTP انجام می‌شود. کاربر انتقال امن credentials لازم MLflow را صریحاً مجاز کرده است.

`verify` تمام مراحل زیر را اجرا می‌کند و با اولین شکست متوقف می‌شود:

1. SHA کل ZIP متادیتا، allowlist دقیق، CRC و hash هر فایل؛ انتشار بدون overwrite مغایرت.
2. SHA کل ZIP خام، CRC تمام اعضا، labels.csv و SHA تمام ۴۵۲۹ فایل؛ سپس حذف **فقط نسخهٔ سرورِ ZIP**. ZIP اصلی سیستم محلی حفظ می‌شود.
3. Python، dependency check، نسخه‌های core، CUDA arithmetic، حافظه/نام 3090، WAV/MP3 واقعی و checkout تمیز.
4. forward واقعی CAM++ و head با صفر backward/optimizer step.
5. run زیرساختی در experiment مستقل: upload/download همهٔ آرتیفکت‌ها با SHA، بازخوانی پارامتر/متریک/tag و وضعیت FINISHED.
6. تطبیق همهٔ شواهد با کد، پکیج‌ها، داده، وزن و میزبان فعلی؛ فقط در این صورت `readiness=ready`.

## MLflow و شواهد هر اجرا

Binding experiment در `artifacts/infrastructure/mlflow_state.json` ذخیره می‌شود؛ نام Default و experiment قدیمی بدون binding همین پروژه رد می‌شوند. پارامتر `DAGSHUB_EXPRIMENT_NAME` قدیمی عمداً استفاده نمی‌شود.

هر run شامل config حل‌شده، پارامترها، seed، fingerprint ورودی‌ها، نسخه‌های محیط، Git SHA، snapshot قطعی تمام `src` و manifest آن، گزارش JSON/Markdown، متریک‌ها و آرتیفکت‌های مربوط است. اجراهای مدل parent و fold child دارند؛ prediction، احتمال ۴۴۷کلاسه، gallery، خطاهای هر کلاس و slice، نمودارها و checkpoint/resume ثبت می‌شوند. صف محلی رخدادها در قطعی ارتباط حفظ می‌شود و تحویل تأییدنشده موفق گزارش نمی‌شود.

نام run محلی زیرساخت شامل `local` است. موفقیت آن نشان‌دهندهٔ دسترسی سیستم محلی به MLflow است؛ همین probe باید روی خود سرور تکرار شود. هیچ نمرهٔ مسابقه یا loss آموزشی برای probe ساختگی ثبت نمی‌شود.

## فرمان آیندهٔ شروع

فقط بعد از `readiness=ready` و پیام صریح بعدی کاربر، روی سرور:

```bash
cd /workspace/Speaker-identification-2
export VAST_INSTANCE_ID=50079023
.venv/bin/python scripts/infra/with_project_env.py .venv/bin/python scripts/train.py --config configs/train/campp_baseline.json --execute-training
```

این فرمان اکنون اجرا نشده است. `scripts/train.py` بدون فلگ فقط قراردادها را می‌سنجد. شروع واقعی همهٔ صوت‌ها را دوباره hash می‌کند و پیش از هر استخراج/fit یک roundtrip زندهٔ تازهٔ MLflow می‌خواهد. تغییر config برای F001 نیازمند probe/readiness متناظر همان قرارداد است.
