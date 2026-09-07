# آماده‌سازی CAM++ و زیرساخت آموزش

تاریخ: ۲۰۲۶-۰۹-۰۷. **زیرساخت آماده است و منتظر دستور صریح کاربر برای شروع است. هیچ آموزش، optimizer step یا استخراج سراسری embedding اجرا نشده است.**

## وضعیت واقعی

- اولین بررسی یکپارچه در `2026-09-07T13:12:30Z` روی commit `0bc18fe` به `readiness=ready` رسید: هر ۱۲ بررسی موفق و مانع صفر بود. Git SHA و run دقیق آخرین استقرار از شواهد نسخهٔ جاری خوانده می‌شوند؛ پس از هر commit بررسی مجدد انجام می‌شود.
- مدل اصلی CAM++ است. forward واقعی نهایی روی RTX 3090 در ۱٫۳۷ ثانیه موفق شد: بردار واحد ۵۱۲بعدی و logits با شکل `2×446`؛ backward و optimizer step هر دو صفر بودند.
- ZIP محلی به‌طور کامل بررسی شد: CRC هر ۴۵۳۰ عضو، SHA تمام ۴۵۲۹ صوت و labels، و تمام فایل‌های استخراج‌شده تطبیق داشتند. ZIP اصلی حفظ شد؛ گزارش `artifacts/infrastructure/data_local_verification.json` تنها شاهد بررسی محلی است.
- ۱۰۵ آزمون کامل و بررسی انتهابه‌انتهای probe با backend آزمایشی گذشته‌اند. dry-run قرارداد ۴۵۲۹ فایل و دو fold را تأیید کرده است.
- instance `50079023` اکنون **running** است و SSH مستقیم با کلید RSA موجود تأیید شده است. کمبود ظرفیت اولیه برطرف شده؛ همان instance استفاده می‌شود و هیچ جایگزینی اجاره یا instance دیگری حذف نشده است.
- محیط train قفل‌شده با **۸۲ پکیج** نصب است. runtime نهایی شامل CUDA، `pip check`، نسخه‌های core و decode ترکیب‌های واقعی WAV/MP3 موجود در داده موفق شد.
- هر چهار فایل متادیتا روی سرور با SHA تأیید شده‌اند و ZIP متادیتای منتقل‌شده، پس از تأیید نصب، از سرور حذف شده است. credentials لازم MLflow منتقل شده و فایل `.env` سرور mode برابر `600` دارد.
- experiment مستقل `iaaa2026-campp-infrastructure-20260907` با ID برابر `1` فعال است. نخستین probe یکپارچهٔ موفق `32aabecc021146feae737f808d39ced6` با وضعیت FINISHED، ۱۰ artifact با SHA، ۴۵ پارامتر، ۸ متریک و ۹ tag را بازخوانی کرد؛ گزارش‌های `checks/data.json`، `checks/runtime.json` و `checks/campp.json` داخل همان run هستند.
- آرشیو رسمی کامل دریافت و محتوایش تأیید شد: CRC هر ۴۵۳۰ عضو، SHA/اندازهٔ تمام ۴۵۲۹ صوت و labels بایت‌به‌بایت، مجموعاً ۱۶٬۸۶۱٬۶۸۵٬۱۷۴ بایت در ۱۳۱٫۸۹ ثانیه. ZIP رسمی و باقی‌مانده‌های انتقال قبلی سپس حذف شدند؛ دانلود قدیمی خاتمه یافت و نبود فرایندش بررسی شد. ZIP اصلی محلی حفظ شده و حدود ۱۰۳GiB فضای آزاد روی سرور مشاهده شد.
- سرویس Supervisor با نام `speaker_id_campp_b001` از طریق `scripts/infra/install_supervisor.sh` ثبت شده و وضعیت **STOPPED** آن تأیید شده است. `autostart=false` و `autorestart=false` هستند؛ سرویس شروع نشده و منتظر دستور صریح کاربر می‌ماند.

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

محیط train: Python 3.12، torch/torchaudio `2.10.0+cu128`، NumPy `2.2.6`، SciPy `1.15.3`، SoundFile `0.13.1`، MLflow client (`mlflow-skinny`) `3.7.0`، matplotlib `3.11.1`. کل dependency graph در `uv.lock` قفل شده است. `bootstrap_server.sh` با uv `0.11.28` همین محیط ۸۲پکیجی را روی سرور نصب کرده است. مقایسهٔ نسخه‌های core با بازه‌های راهنمای مسابقه در probe اولیه موفق بوده؛ فایل راهنما freeze دقیق سرور نیست و آزمون بستهٔ نهایی آفلاین همچنان مرحلهٔ تحویل خواهد بود.

## چرخهٔ استقرار و بررسی

نصب نخستین انجام شده است: محیط با `bootstrap_server.sh`، دریافت رسمی با `download_competition.sh` و ثبت سرویس متوقف با `install_supervisor.sh`. هویت دانلود در `configs/infra/archive_sources.json` ثبت است. برای بررسی عادی پس از commit/push، نصب یا بارگذاری دوبارهٔ داده لازم نیست؛ از ریشهٔ workspace محلی:

```powershell
.venv/Scripts/python.exe scripts/infra/vast_control.py show
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py deploy --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py verify --archive-source official_original --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py download-evidence --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
```

SSH مستقیم با کلید RSA ثبت‌شدهٔ حساب موفق شد؛ hostname برابر `4593b1f57a8e`، Python سیستم `3.12.3`، GPU برابر RTX 3090 با ۲۴GiB و driver `580.95.05` مشاهده شدند. پراکسی SSH اتصال را بست و ED25519 در حساب ثبت نبود؛ بعد از خواندن لاگ و تطبیق کلیدهای عمومی، RSA موجود انتخاب شد. هیچ کلید جدیدی اضافه نشد. در صورت خطای SSH، ابتدا `vast_control.py logs` خوانده می‌شود و علت تعیین می‌شود؛ کلید یا تنظیمات کورکورانه تغییر نمی‌کنند.

`upload-assets` فقط متادیتا، وزن عمومی، binding آزمایش جدید، ZIP خام و سه متغیر MLflow را انتقال می‌دهد. tokenهای Vast/Git روی سیستم محلی باقی می‌مانند. فایل `.env` سرور regular file با mode `600` است؛ انتقال رمزگذاری‌شده با SSH/SFTP انجام می‌شود. کاربر انتقال امن credentials لازم MLflow را صریحاً مجاز کرده است.

انتقال SFTP مستقیم و پراکسی هر دو حدود ۳۰ تا ۴۰KB/s بودند؛ دریافت از URL رسمی با aria2 و میانگین حدود ۳۲MiB/s کامل شد. باقی‌مانده‌های انتقال‌های قبلی پس از تأیید کامل داده حذف شدند.

آرشیو [منبع رسمی مسابقه](https://iaaa-contest-speaker.s3.ir-thr-at1.arvanstorage.ir/iaaa-contest-speaker.zip?versionId=) اندازهٔ ۹٬۷۶۴٬۹۵۰٬۶۰۳ بایت و ریشهٔ `training/` داشت. SHA اندازه‌گیری‌شدهٔ دانلود `6ca989a80fa36030ebe7fb3f17e5299be5adf8221617e08eeb6956ded68bc9c4` است؛ checksum منتشرشدهٔ مستقلی از برگزارکننده در اختیار نداریم. صحت محتوا با manifest قبلی تمام صوت‌ها و SHA بایت‌به‌بایت labels محلی تأیید شد. بسته‌بندی با ZIP محلی متفاوت است و SHA canonical در `configs/infra/deployment.json` بدون تغییر مانده است.

`verify` تمام مراحل زیر را اجرا می‌کند و با اولین شکست متوقف می‌شود:

1. SHA کل ZIP متادیتا، allowlist دقیق، CRC و hash هر فایل؛ انتشار بدون overwrite مغایرت.
2. تطابق ZIP انتخاب‌شده با هویت دانلود ثبت‌شده، ریشهٔ مجاز `training/`، CRC تمام اعضا، SHA/اندازهٔ تمام ۴۵۲۹ صوت و SHA بایت‌به‌بایت labels محلی؛ سپس حذف **فقط ZIP دریافت‌شدهٔ سرور پس از تأیید**. ZIP اصلی سیستم محلی حفظ می‌شود.
3. Python، dependency check، نسخه‌های core، CUDA arithmetic، حافظه/نام 3090، WAV/MP3 واقعی و checkout تمیز.
4. forward واقعی CAM++ و head با صفر backward/optimizer step.
5. run زیرساختی در experiment مستقل: upload/download همهٔ آرتیفکت‌ها با SHA، بازخوانی پارامتر/متریک/tag و وضعیت FINISHED.
6. تطبیق همهٔ شواهد با کد، پکیج‌ها، داده، وزن و میزبان فعلی؛ فقط در این صورت `readiness=ready`.

## MLflow و شواهد هر اجرا

Binding experiment در `artifacts/infrastructure/mlflow_state.json` ذخیره می‌شود؛ نام Default و experiment قدیمی بدون binding همین پروژه رد می‌شوند. پارامتر `DAGSHUB_EXPRIMENT_NAME` قدیمی عمداً استفاده نمی‌شود.

هر run شامل config حل‌شده، پارامترها، seed، fingerprint ورودی‌ها، نسخه‌های محیط، Git SHA، snapshot قطعی تمام `src` و manifest آن، گزارش JSON/Markdown، متریک‌ها و آرتیفکت‌های مربوط است. اجراهای مدل parent و fold child دارند؛ prediction، احتمال ۴۴۷کلاسه، gallery، خطاهای هر کلاس و slice، نمودارها و checkpoint/resume ثبت می‌شوند. صف محلی رخدادها در قطعی ارتباط حفظ می‌شود و تحویل تأییدنشده موفق گزارش نمی‌شود.

run محلی `cfb81b7b9bbd4720865780247c53a3e9` و probe ارتباط سرور `fa2f0afa3db6479b919544d527f0796a` به‌عنوان سابقه حفظ شده‌اند. مرجع آخرین Git SHA و run متناظر، گزارش‌های شواهد است. محل تحویل نسخهٔ محلی آن‌ها `artifacts/infrastructure/server_evidence/readiness.json` و `artifacts/infrastructure/server_evidence/mlflow_preflight_result.json` است؛ نتیجهٔ آخرین `download-evidence` مرجع به‌روز این فایل‌هاست. شناسهٔ اولین probe یکپارچه، جای شناسهٔ آخرین استقرار را نمی‌گیرد.

## فرمان آیندهٔ شروع

پیکربندی نسخه‌دار در `configs/infra/supervisor_campp_baseline.conf` و wrapper در `scripts/infra/run_campp_baseline.sh` قرار دارند. wrapper محیط MLflow و شناسهٔ instance را بارگذاری می‌کند و اجرای مدل را از اتصال SSH مستقل نگه می‌دارد. نصب با اسکریپت نسخه‌دار انجام شده، تطابق config نصب‌شده کنترل شده و سرویس اکنون STOPPED است.

فقط بعد از `readiness=ready` و پیام صریح بعدی کاربر، روی سرور:

```bash
cd /workspace/Speaker-identification-2
supervisorctl start speaker_id_campp_b001
```

این فرمان اکنون اجرا نشده است. `scripts/train.py` بدون فلگ فقط قراردادها را می‌سنجد. شروع واقعی از طریق wrapper، همهٔ صوت‌ها را دوباره hash می‌کند و پیش از هر استخراج/fit یک roundtrip زندهٔ تازهٔ MLflow می‌خواهد. تغییر config برای F001 نیازمند probe/readiness متناظر همان قرارداد است.

## اصلاح حاصل از آزمایش واقعی MLflow

مسیر HTTP مربوط به دانلود آرتیفکت `tar.gz`، محتوای gzip را خودکار باز می‌کرد: ۱۱۶٬۵۲۸ بایت ارسال‌شده به ۴۶۰٬۸۰۰ بایت tar تبدیل می‌شد. بررسی نشان داد خروجی دقیقاً tar بازشده است، ولی شرط SHA بایت‌به‌بایت به‌درستی آن را رد کرد. قالب snapshot به ZIP قطعی تغییر کرد؛ شرط صحت SHA تضعیف نمی‌شود. run نخست `03f4686ad2d246dcad861935c746e0b3` برای سابقهٔ خطا با وضعیت FAILED حفظ شده است. UTF-8 خروجی CLI ویندوز نیز صریح تنظیم شد تا چاپ لینک‌های SDK مانع ثبت وضعیت نهایی نشود.

پارامترها در batchهای حداکثر ۱۰۰تایی و متریک‌ها در batchهای حداکثر ۵۰۰تایی ارسال می‌شوند. شکست بخشی از ارسال، cursor رخداد تأییدنشده را جلو نمی‌برد؛ ارسال مجدد پس از قطعی ممکن است یک metric را تکرار کند ولی آن را از دست نمی‌دهد.
