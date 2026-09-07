# وزن‌ها و محیط بازتولید ممیزی مدل‌های عمومی

در EDA از وزن‌های عمومی و ثابت ECAPA، AST و Whisper base استفاده شد. هیچ وزن آموزش‌دیده روی مسابقه از پروژهٔ قبلی وارد نشده است. فایل‌های واقعاً استفاده‌شده، همراه تنظیمات، tokenizer و metadata عمومی نسخهٔ دانلود، در این پروژه قرار گرفته‌اند:

| مدل | مسیر قابل انتقال |
| --- | --- |
| SpeechBrain ECAPA | `artifacts/models/ecapa/` |
| MIT AST AudioSet | `artifacts/models/ast/` |
| OpenAI Whisper base | `artifacts/models/whisper_base/` |

فهرست دقیق فایل‌ها، نسخهٔ مخزن، SHA256 و نتیجهٔ تأیید کپی در `reports/eda/model_assets.json` است. این پوشه‌ها در Git نادیده گرفته می‌شوند؛ هنگام انتقال پروژه باید صریحاً کپی شوند. وجود registry به‌تنهایی به معنی انتقال وزن‌ها نیست. برای تأیید دوباره، بدون دسترسی به پروژهٔ قبلی یا cache خارجی:

```powershell
.venv\Scripts\python.exe scripts\eda\stage_models.py --verify-only
```

مسیرهای قبلی در `embedding_summary.json` و `semantic_summary.json` تغییر نکرده‌اند؛ آن‌ها سابقهٔ واقعی اجرای انجام‌شده هستند. registry، محل مستقل فعلی همان فایل‌ها را با تطابق بایت‌به‌بایت نشان می‌دهد. کپی وزن‌ها به معنی اجرای دوبارهٔ inference در محیط جدید نیست.

## محیط واقعاً استفاده‌شده

اجرای ممیزی از Python 3.12.13 پروژهٔ فعلی و مسیر read-only پکیج‌های نصب‌شدهٔ پروژهٔ قبلی استفاده کرد. نسخه‌های مستقیم ثبت‌شده شامل PyTorch و Torchaudio `2.11.0+cu126`، SpeechBrain `1.1.0`، Transformers `4.57.6`، NumPy `2.4.6`، SciPy `1.18.0` و SoundFile `0.14.0` هستند. نسخهٔ libsndfile گزارش‌شده `1.2.2` است. فهرست metadata پکیج‌های موجود در آن محیط در `reports/eda/model_runtime_inventory.json` ذخیره شده است؛ این فهرست شامل پکیج‌های استفاده‌نشده هم هست و فایل lock یا اثبات بازتولید نصب نیست.

**این محیط پژوهشی با محیط لیدربرد یکسان نیست.** NumPy، SciPy و SoundFile این اجرا بیرون از بازه‌های فایل راهنمای لیدربرد هستند. آزمون inference در محیط واقعی لیدربرد یا محیط Linux مطابق آن هنوز انجام نشده است. نتایج EDA معتبر بودن قرارداد ارسال و وابستگی‌های submission را تأیید نمی‌کنند. فایل محیط اصلی پروژه و `uv.lock` برای این کار عوض نشده‌اند.

## دستور پیشنهادی ساخت محیط پژوهشی مستقل

دستورهای زیر فقط دستور بازسازی پیشنهادی‌اند؛ در این مرحله نصب نشده‌اند و حل کامل وابستگی‌ها در محیط تمیز آزمایش نشده است. نسخه‌ها از فایل‌های واقعاً نصب‌شده استخراج شده‌اند، اما موجود بودن فعلی wheel مناسب برای هر سیستم‌عامل یا سازگاری مجموعهٔ ترانزیتیو در نصب تازه بررسی نشده است. برای حفظ محیط فعلی، محیط تازه در مسیر ignored زیر `artifacts` ساخته می‌شود:

```powershell
.venv\Scripts\python.exe -m venv artifacts\tooling\model-research-venv
artifacts\tooling\model-research-venv\Scripts\python.exe -m pip install torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
artifacts\tooling\model-research-venv\Scripts\python.exe -m pip install -r configs\eda\model_audit_requirements.txt
artifacts\tooling\model-research-venv\Scripts\python.exe -m pip check
```

این نصب نیازمند دسترسی به مخازن پکیج است؛ برای یک ماشین آفلاین باید wheelهای دقیق روی ماشین متصل آماده شوند و پیش از انتقال، نصب تمیز و اجرای نمونه بررسی شود. چنین wheelhouse یا آزمونی در این مرحله ساخته نشده است. `configs/eda/model_audit_requirements.txt` فقط pin وابستگی‌های مستقیم پژوهشی را ثبت می‌کند؛ قفل کامل ترانزیتیو با hash نیست.

پس از آماده و تأیید شدن محیط مستقل، اجرای ممیزی با وزن‌های موجود در همین پروژه و بدون `--extra-site-packages` امکان‌پذیر است. دستورهای زیر برای بازسازی خروجی اصلی هستند و همان مسیرهای گزارش را می‌نویسند؛ این اجرا در این مرحله تکرار نشده است:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
artifacts\tooling\model-research-venv\Scripts\python.exe scripts\eda\audit_embeddings.py --model-dir artifacts\models\ecapa
artifacts\tooling\model-research-venv\Scripts\python.exe scripts\eda\audit_semantic.py --ast artifacts\models\ast --whisper artifacts\models\whisper_base
```

نمودارهای EDA و گزارش نهایی همچنان با محیط سبک اصلی پروژه تولید می‌شوند. انتقال وزن‌های عمومی، اجرای ممیزی پژوهشی، آماده‌سازی آموزش روی 3090 و سازگاری بستهٔ لیدربرد چهار کار جدا هستند؛ دو کار آخر هنوز شروع نشده‌اند.
