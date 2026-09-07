# آماده‌سازی CAM++ و زیرساخت آموزش

تاریخ: ۲۰۲۶-۰۹-۰۷. **هیچ آموزش، optimizer step یا استخراج سراسری embedding در این مرحله اجرا نشده است.**

## وضعیت واقعی

- مدل اصلی CAM++ است. وزن عمومی VoxCeleb و معماری رسمی نسخه‌دار آماده‌اند؛ probe محلی روی فایل واقعی، embedding نرمال‌شدهٔ ۵۱۲بعدی و graph خروجی ۴۴۶کلاسه را تأیید کرده است. این probe روی CPU با runtime پژوهشی انجام شده و تأیید سازگاری CUDA/لیدربرد نیست.
- ZIP محلی به‌طور کامل بررسی شد: CRC هر ۴۵۳۰ عضو، SHA تمام ۴۵۲۹ صوت و labels، و تمام فایل‌های استخراج‌شده تطبیق داشتند. ZIP اصلی حفظ شد؛ گزارش `artifacts/infrastructure/data_local_verification.json` تنها شاهد بررسی محلی است.
- آزمون‌های داده، نقش‌های کالیبراسیون، امتیازدهی، CAM++، انتقال ZIP، tracking و readiness گذشته‌اند. dry-run قرارداد ۴۵۲۹ فایل و دو fold را تأیید کرده است.
- instance `50079023` اکنون **running** است و SSH مستقیم با کلید RSA موجود تأیید شده است. کمبود ظرفیت اولیه برطرف شده؛ همان instance استفاده می‌شود و هیچ جایگزینی اجاره یا instance دیگری حذف نشده است.
- محیط train قفل‌شده با **۸۲ پکیج** روی سرور نصب شده است. بررسی رسمی CUDA، forward مدل CAM++ و جمع‌بندی readiness هنوز باقی است.
- هر چهار فایل متادیتا روی سرور با SHA تأیید شده‌اند و ZIP متادیتای منتقل‌شده، پس از تأیید نصب، از سرور حذف شده است. credentials لازم MLflow منتقل شده و فایل `.env` سرور mode برابر `600` دارد.
- experiment مستقل `iaaa2026-campp-infrastructure-20260907` با ID برابر `1` فعال است. probe ارتباط و ثبت MLflow **روی خود سرور** با run `fa2f0afa3db6479b919544d527f0796a` به وضعیت FINISHED رسیده: ۷ آرتیفکت با SHA، ۴۵ پارامتر، ۵ متریک و ۹ tag بازخوانی و تأیید شده‌اند.
- در زمان این به‌روزرسانی، انتقال ZIP خام حدود ۱۰GB ادامه دارد. تأیید کامل آرشیو و تمام فایل‌های خام روی سرور، حذف نسخهٔ منتقل‌شده پس از تأیید، probeهای CUDA/CAM++ و جمع‌بندی نهایی readiness انجام نشده‌اند. **آمادگی کامل برای شروع آموزش هنوز اعلام نشده است.**
- پیکربندی Supervisor برای `speaker_id_campp_b001` با `autostart=false` و `autorestart=false` آماده است. نصب و ثبت سرویس و مشاهدهٔ وضعیت STOPPED در این به‌روزرسانی هنوز تأیید نشده‌اند؛ هدف این است که سرویس تا دستور صریح کاربر متوقف بماند.

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

محیط train: Python 3.12، torch/torchaudio `2.10.0+cu128`، NumPy `2.2.6`، SciPy `1.15.3`، SoundFile `0.13.1`، MLflow client (`mlflow-skinny`) `3.7.0`، matplotlib `3.11.1`. کل dependency graph در `uv.lock` قفل شده است. `bootstrap_server.sh` با uv `0.11.28` همین محیط ۸۲پکیجی را روی سرور نصب کرده است. نسخه‌های core در مرحلهٔ runtime با بازه‌های راهنمای مسابقه مقایسه می‌شوند؛ فایل راهنما freeze دقیق سرور نیست و آزمون بستهٔ نهایی آفلاین همچنان مرحلهٔ تحویل خواهد بود.

## چرخهٔ استقرار و بررسی

فرمان‌های چرخهٔ استقرار از ریشهٔ workspace محلی، با کلید SSH ثبت‌شدهٔ حساب، به شکل زیرند. نصب محیط و متادیتا انجام شده و انتقال ZIP خام در جریان است؛ برای ادامهٔ مرحلهٔ جاری، پس از پایان انتقال نوبت `verify` و سپس دریافت شواهد است. instance اکنون running است و درخواست start مجدد لازم نیست.

```powershell
uv --cache-dir artifacts/tooling/uv-cache sync --locked --group ops --group eda
.venv/Scripts/python.exe scripts/infra/vast_control.py show
.venv/Scripts/python.exe scripts/infra/prepare_transfer.py
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py inspect --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py deploy --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py bootstrap --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py upload-assets --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py verify --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
.venv/Scripts/python.exe scripts/infra/deploy_workspace.py download-evidence --identity-file C:/Users/AmirhosseinEsbati/.ssh/id_rsa
```

SSH مستقیم با کلید RSA ثبت‌شدهٔ حساب موفق شد؛ hostname برابر `4593b1f57a8e`، Python سیستم `3.12.3`، GPU برابر RTX 3090 با ۲۴GiB و driver `580.95.05` مشاهده شدند. پراکسی SSH اتصال را بست و ED25519 در حساب ثبت نبود؛ بعد از خواندن لاگ و تطبیق کلیدهای عمومی، RSA موجود انتخاب شد. هیچ کلید جدیدی اضافه نشد. در صورت خطای SSH، ابتدا `vast_control.py logs` خوانده می‌شود و علت تعیین می‌شود؛ کلید یا تنظیمات کورکورانه تغییر نمی‌کنند.

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

run محلی تاریخی `cfb81b7b9bbd4720865780247c53a3e9` با وضعیت FINISHED، دسترسی سیستم محلی را تأیید کرده است؛ نتیجه در `artifacts/infrastructure/mlflow_local_probe_zip/preflight_result.json` قرار دارد. probe ارتباط سرور نیز اکنون با run `fa2f0afa3db6479b919544d527f0796a` موفق شده است. مرحلهٔ `verify` شواهد نهایی کد، مدل، داده و MLflow را با قرارداد فعلی تطبیق خواهد داد. هیچ نمرهٔ مسابقه یا loss آموزشی برای probe ساختگی ثبت نمی‌شود.

## فرمان آیندهٔ شروع

پیکربندی نسخه‌دار در `configs/infra/supervisor_campp_baseline.conf` و wrapper در `scripts/infra/run_campp_baseline.sh` قرار دارند. wrapper محیط MLflow و شناسهٔ instance را بارگذاری می‌کند و اجرای مدل را از اتصال SSH مستقل نگه می‌دارد. ثبت موفق سرویس باید آن را در وضعیت STOPPED نگه دارد؛ هنوز تأیید این نصب در شواهد این به‌روزرسانی ثبت نشده است.

فقط بعد از `readiness=ready`، تأیید ثبت سرویس در وضعیت STOPPED و پیام صریح بعدی کاربر، روی سرور:

```bash
cd /workspace/Speaker-identification-2
supervisorctl start speaker_id_campp_b001
```

این فرمان اکنون اجرا نشده است. `scripts/train.py` بدون فلگ فقط قراردادها را می‌سنجد. شروع واقعی از طریق wrapper، همهٔ صوت‌ها را دوباره hash می‌کند و پیش از هر استخراج/fit یک roundtrip زندهٔ تازهٔ MLflow می‌خواهد. تغییر config برای F001 نیازمند probe/readiness متناظر همان قرارداد است.

## اصلاح حاصل از آزمایش واقعی MLflow

مسیر HTTP مربوط به دانلود آرتیفکت `tar.gz`، محتوای gzip را خودکار باز می‌کرد: ۱۱۶٬۵۲۸ بایت ارسال‌شده به ۴۶۰٬۸۰۰ بایت tar تبدیل می‌شد. بررسی نشان داد خروجی دقیقاً tar بازشده است، ولی شرط SHA بایت‌به‌بایت به‌درستی آن را رد کرد. قالب snapshot به ZIP قطعی تغییر کرد؛ شرط صحت SHA تضعیف نمی‌شود. run نخست `03f4686ad2d246dcad861935c746e0b3` برای سابقهٔ خطا با وضعیت FAILED حفظ شده است. UTF-8 خروجی CLI ویندوز نیز صریح تنظیم شد تا چاپ لینک‌های SDK مانع ثبت وضعیت نهایی نشود.

پارامترها در batchهای حداکثر ۱۰۰تایی و متریک‌ها در batchهای حداکثر ۵۰۰تایی ارسال می‌شوند. شکست بخشی از ارسال، cursor رخداد تأییدنشده را جلو نمی‌برد؛ ارسال مجدد پس از قطعی ممکن است یک metric را تکرار کند ولی آن را از دست نمی‌دهد.
