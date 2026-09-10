# سیاست full-enrollment برای فیت نهایی CAM++

## تصمیم

در توسعه و مقایسهٔ آزمایش‌ها، `calibration_roles.csv` و گروه‌های بیرونی ثابت می‌مانند. برای هر کلاس known که حداقل دو گروه محتوایی دارد، یک گروه مستقل به‌عنوان `known_calibration_query` نگه داشته می‌شود. این گروه در آموزش encoder، گالری مرجع و انتخاب threshold همان fold وارد نمی‌شود.

پس از قفل‌شدن معماری، تعداد updateها و decoder، فیت نهایی با تمام ردیف‌های `train_eligible` انجام می‌شود؛ یعنی گروه‌های calibration نیز به آموزش encoder و گالری enrollment برمی‌گردند. این refit مدل نهایی است و به‌عنوان OOF جدید گزارش نمی‌شود.

## دلیل آماری

این query مستقل برای اندازه‌گیری خطای `unknown→known` و `known→unknown` و تنظیم ردکردن unknown لازم است. استفادهٔ هم‌زمان از یک ردیف برای آموزش encoder و تنظیم threshold، امتیاز توسعه را خوش‌بینانه می‌کند و به‌خصوص در مسئلهٔ open-set می‌تواند روی لیدربرد افت ایجاد کند.

در split منجمد فعلی، fold صفر ۴۴۳ ردیف known calibration و ۶۶۲ ردیف known enrollment دارد؛ fold یک ۴۴۵ و ۶۶۷ ردیف دارد. بنابراین refit نهایی تقریباً ۴۰٪ پشتیبانی known در بخش داخلی هر fold را بازیابی می‌کند. این اعداد ردیف‌اند؛ یک گروه محتوایی ممکن است چند فایل تکراری داشته باشد.

## قرارداد کالیبراسیون فیت نهایی

برای encoder فریز‌شده، `all_training_crossfit_scores` هر query را با حذف کامل گروه محتوایی‌اش امتیازدهی می‌کند و سپس تمام referenceهای معتبر را به گالری نهایی برمی‌گرداند. برای encoder سازگارشده، threshold نباید از پیش‌بینی in-sample refit انتخاب شود. یکی از این دو مسیر باید پیشاپیش قفل شود:

1. cross-fitting چند encoder که هر query را از fit خود حذف می‌کند؛ یا
2. نگه‌داشتن calibration مستقل در مرحلهٔ توسعه، فریزکردن threshold/temperature و سپس audit تغییر توزیع امتیاز پس از refit.

هر full-enrollment candidate فقط در صورتی قابل promotion است که روی outerهای دست‌نخورده حداقل `+0.0015` Macro-F1 بدهد، در هیچ fold افت معنی‌دار نداشته باشد و تعداد خطاهای `unknown→known` را بدتر نکند. امتیاز training calibration به‌تنهایی معیار promotion نیست.

## پیاده‌سازی موجود

مسیر فیت نهایی frozen در `src/speaker_id/training/final_references.py` پیاده شده و تست‌های `tests/test_final_references.py` حذف کامل گروه query، بازگرداندن تمام referenceها و رفتار singleton را پوشش می‌دهند.
