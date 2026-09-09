# F005: سازگاری جفت نمای کوتاه و بلند برای CAM++ advanced

F005 فقط endpoint عمومی advanced CAM++ با خروجی خام ۱۹۲بعدی و وزن
`campplus_cn_en_common.pt` با SHA256 ثبت‌شده در config را آموزش می‌دهد. مدل عمومی
۵۱۲بعدی در تمام مراحل ثابت می‌ماند و فقط در scoring پس از آموزش استفاده می‌شود.

در هر outer fold، مرحلهٔ ۶۰۰گامی head-only دقیقاً یک‌بار اجرا می‌شود. checkpoint
همان مرحله، با هش بایت و state یکسان، نقطهٔ شروع چهار tail پانصدگامی است: یک control
با dual-view AAM و سه treatment با ضریب cosine ثابت ۰٫۵ و ضرایب MSE برابر صفر،
۰٫۱ و ۰٫۵. خروجی خام `h` همان خروجی نهایی ۱۹۲بعدی CAM++ پیش از L2-normalization
است. target نمای بلند فقط در دو loss سازگاری stop-gradient دارد؛ long-AAM همچنان
از نمای بلند به encoder گرادیان می‌دهد.

ضریب‌های consistency در ۱۰۰ گام نخست tail خطی ramp می‌شوند: در global step
صفرمبنای ۶۰۰، scale صفر است؛ در step ۶۹۹ به یک می‌رسد و تا پایان ثابت می‌ماند.
بنابراین ضریب مؤثر cosine و MSE برابر ضریب هر arm ضرب‌در همین scale است و control
در تمام گام‌ها دقیقاً ضریب صفر دارد. scale و هر دو ضریب مؤثر در history و MLflow
ثبت می‌شوند.

انتخاب arm برای هر outer fold جدا انجام می‌شود. فقط queryهای known از نقش‌های
کالیبراسیون اصلی و مستقل استفاده می‌شوند و gallery شناخته‌شده تمام referenceهای
مجاز بخش آموزشی را با حذف کل content group همان query به کار می‌گیرد. پوشش واقعی
known در foldهای صفر و یک به‌ترتیب ۴۴۳ و ۴۴۵ کلاس است و ۳ و ۱ کلاس غایب صریحاً
گزارش می‌شوند. control نیز selectable است؛ برنده‌شدن آن یعنی شواهد این آزمایش از
loss سازگاری پشتیبانی نمی‌کند. تا پیش از ذخیره و بازخوانی seal انتخاب، هیچ query یا
cohort مربوط به unknown و هیچ outer label مصرف نمی‌شود.

پس از seal، scoring کامل با `heldout_reference_scores` انجام می‌شود. سه مقایسهٔ
اصلی عبارت‌اند از frozen public+advanced در همین پروتکل، control آموزش‌دیده در همین
پروتکل و arm منتخب. C002b معیار تاریخی و بهترین incumbent فعلی است؛ چون پروتکل
توسعه‌ای یکسان نیست، F005 ادعای بازتولید دقیق C002b ندارد. ارتقای treatment به حداقل
۰٫۰۰۳ Macro-F1 بهتر از control تازه و C002b نیاز دارد. اگر control منتخب باشد، برای
ارتقا باید حداقل ۰٫۰۰۳ از C002b بهتر باشد. شرط‌های fold، accuracy، short-known top-1،
U→K، K→K و paired whole-content-group bootstrap نیز هم‌زمان اعمال می‌شوند. گروه
bootstrap یک واحد تجزیه‌ناپذیر است و می‌تواند چند true label داشته باشد.

## دستورات امن پیش از اجرا

اعتبارسنجی فقط config و بدون دسترسی به داده، CUDA یا MLflow:

```bash
.venv/bin/python scripts/train_f005.py
```

اعتبارسنجی role/data و ساخت plan بدون آموزش:

```bash
scripts/infra/run_f005.sh --validate-data --verify-audio --verify-sources --emit-complete-plan-hashes
```

probe محدود یک backward در FP32، بدون optimizer step:

```bash
scripts/infra/run_f005.sh --execute-probe --outer-fold 0
```

launcher مقدار `CUBLAS_WORKSPACE_CONFIG=:4096:8`، marker ثابت
`VAST_INSTANCE_ID=50288952` و سه متغیر thread را پیش از ورود Python تنظیم می‌کند.
این marker با readiness و پیکربندی Supervisor یکسان است. worker از
deterministic-algorithms با حالت خطا استفاده
می‌کند؛ warning قابل ادامه پذیرفته نمی‌شود. خود worker نیز مستقل از orchestrator،
receipt واقعی `audio_hashes_checked=True` و تطابق `VAST_INSTANCE_ID` با readiness
را پیش از ساخت Torch/model الزام می‌کند.

پس از موفقیت probe، launcher کامل فقط با گیت صریح زیر قابل اجراست:

```bash
scripts/infra/install_f005_supervisor.sh
supervisorctl start speaker_id_campp_f005
```

installer فقط job را ثبت و صحت‌سنجی می‌کند و آن را شروع نمی‌کند؛ فرمان دوم شروع
صریح اجرای کامل است.

این launcher یک parent و ده child در MLflow می‌سازد: دو shared-head و هشت tail.
هر دو انتخاب known-only و هر سه policy هر fold پیش از نخستین دسترسی به outer truth
مهر و از دیسک بازخوانی می‌شوند. اجرای قطع‌شده از همان run و checkpoint معتبر ادامه
می‌یابد و stage کامل‌شده دوباره آموزش نمی‌بیند. برای اجرای پس‌زمینه، اسکریپت
`scripts/infra/install_f005_supervisor.sh` job را با `autostart=false` و
`autorestart=false` ثبت می‌کند؛ شروع job همچنان یک عمل دستی و جداست.
فرمان Supervisor مسیر منطقی ثابت `F005_supervised_primary` را استفاده می‌کند.
اجرای نخست آن مسیر را می‌سازد، اجرای دوباره پس از وقفه همان state امضاشده را
resume می‌کند و اجرای دوباره پس از تکمیل فقط نتیجهٔ مهرشده را بازمی‌گرداند؛ در
نتیجه restart دستی نمی‌تواند ناخواسته یک run آموزشی تکراری بسازد. گزینهٔ
`--resume-dir` برای بازیابی صریح اپراتور همچنان موجود است.

هویت spool محلی parent و هر child پیش از نخستین عملیات remote در state پایدار
می‌شود؛ بنابراین خطای شبکه یا قطع provider همان run را باز می‌کند و run تازه‌ای
نمی‌سازد. checkpoint و embedding دارای staging ثابت `.partial` هستند: فایل نهایی
خراب هرگز جایگزین نمی‌شود، checkpoint فقط پس از load کامل encoder، head، optimizer
و RNG پذیرفته می‌شود، و partial مشتق‌شدهٔ ناقص از منبع pin‌شده بازسازی می‌شود.
آخرین خط ناقص history حذف می‌شود و gap، تکرار یا جابه‌جایی stepها fail-closed است.
cacheهای known-score و pretruth که پس از commit فایل و پیش از commit state یتیم
مانده‌اند فقط پس از تطابق هویت و آرایه‌ها بازیابی می‌شوند. شکست parity دقیق
CPU/CUDA پیش از تغییر `outer_truth_materialized` اجرای outer را متوقف می‌کند.

checkpoint، optimizer، embedding، صوت خام و credential هرگز به MLflow فرستاده
نمی‌شوند. checkpointهای میانی، cacheها و armهای ناموفق روی سرور می‌مانند؛ انتقال
محلی فقط بعد از عبور از promotion gate انجام می‌شود. گزارش parent شامل OOF و هر دو
fold، جهت خطا، اختلاف‌ها، bootstrap گروهی، تمام gateها و وضعیت جداگانهٔ هدف ۰٫۹۶۵
است.

امتیاز ۰٫۹۶۵ معیار پایان Goal است. یک run می‌تواند با بهبود معتبر ۰٫۰۰۳ به incumbent
جدید تبدیل شود ولی هنوز Goal را تمام نکند؛ این دو verdict باید جدا ثبت شوند.
