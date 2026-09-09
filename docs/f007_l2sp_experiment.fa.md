# F007: سازگارسازی tail با L2-SP

## دلیل اجرای آزمایش

F005 با scorer نقش‌های مستقل، از C002b عقب ماند؛ اما bridge cache-only نشان داد
که بخش عمدهٔ آن فاصله از calibration/gate می‌آید، نه از embeddingهای tail:

| مقایسه | Macro-F1 OOF |
| --- | ---: |
| C002b تاریخی | ۰٫۹۵۶۵۲۸۲۸۹ |
| F005 frozen با protocol مستقل | ۰٫۹۵۱۱۲۳۴۱۳ |
| F005 control با policy ثابت تاریخی C002b، فقط تشخیصی | ۰٫۹۵۶۰۰۱۵۸۴ |

اجرای bridge، C002b را دقیقاً بازپخش کرد. control فقط ۷ فایل را بهتر و ۷
فایل را بدتر کرد و accuracy آن با C002b برابر بود. استفاده از policy تاریخی
C002b برای مدل adapted معتبر نیست، زیرا بخشی از calibration تاریخی در fit
encoder آن مدل بوده است. بنابراین عدد bridge فقط تشخیصی است.

## فرضیه

در F005، tail control اندکی از representation پایه دور شده و با scorer مستقل
ضعیف شده است. L2-SP این drift را محدود می‌کند، در حالی که AAM روی کوتاه و بلند
همچنان امکان سازگارسازی مفید را دارد:

\[
L = \frac{L_{AAM}(x_{short}) + L_{AAM}(x_{long})}{2}
+ \lambda \frac{1}{2}\sum_{p \in \mathcal P}\|p-p_0\|_2^2
\]

\(p_0\) snapshot encoder پس از shared-head و پیش از tail است. مجموعهٔ
\(\mathcal P\) فقط parameterهای trainable encoder خارج از BatchNorm است؛
head AAM و BatchNorm affine جریمه نمی‌شوند.

## بازوها و بودجه

کنترل F005 همان checkpoint و cache کامل موجود را reuse می‌کند. فقط این چهار
tail جدید اجرا می‌شوند:

| fold | بازو | λ |
| ---: | --- | ---: |
| ۰ | `l2sp_001` | ۰٫۰۱ |
| ۰ | `l2sp_01` | ۰٫۱ |
| ۱ | `l2sp_001` | ۰٫۰۱ |
| ۱ | `l2sp_01` | ۰٫۱ |

هر بازو از shared-head byte-identical F005، همان seed، crop plan، ۵۰۰ tail
step، optimizer، LR schedule و fit rows استفاده می‌کند. dynamic consistency و
MSE F005 در F007 وجود ندارند.

Gradient L2-SP در شروع tail صفر است؛ به همین علت λ با آن مرحله انتخاب نمی‌شود.
یک probe مجازی و غیرماندگار، اول task-only step 600 را روی clone انجام می‌دهد و
در batch step 601 نسبت \(\|\lambda g_{SP}\|/\|g_{task}\|\) را برای هر λ ثبت
می‌کند. λها از پیش ثابت‌اند و probe بازوی تازه‌ای ایجاد یا انتخاب نمی‌کند.

## ارزیابی معتبر

همان نقش‌های group-disjoint موجود استفاده می‌شوند: برای fold صفر ۴۴۳ query
known و ۵۵۶ query unknown، و برای fold یک ۴۴۵ query known و ۵۵۶ query unknown.
این گروه‌ها با encoder-fit هم‌پوشانی ندارند.

ترتیب اجرا چنین است:

1. control و دو بازو فقط با known calibration query رتبه‌بندی می‌شوند؛ tie-break
   به‌ترتیب control، `l2sp_001` و `l2sp_01` است.
2. بازوی منتخب روی دیسک seal می‌شود.
3. alpha و policy open-set فقط با queryهای مستقل known و unknown fit و seal
   می‌شوند.
4. sealهای هر دو fold reload می‌شوند؛ سپس outer labels هر fold دقیقاً یک بار
   ارزیابی می‌شوند.

C002b با ۰٫۹۵۶۵۲۸۲۸۹ comparator تاریخی باقی می‌ماند. scorer ثابت C002b برای
مدل adapted فقط diagnostic است و هرگز برای selection، promotion یا package
استفاده نمی‌شود.

## شرط promotion

بازوی nonzero باید در هر دو fold انتخاب شود، حداقل ۰٫۰۰۳ از control تازه بهتر
باشد، کران پایین bootstrap گروهی paired آن مثبت باشد، از frozen same-protocol
پایین‌تر نرود، known top-1 و short-known top-1 را کم نکند، و U→K یا K→K را
بدتر نکند. رسیدن به هدف پروژه همچنان به Macro-F1 OOF حداقل ۰٫۹۶۵ با همهٔ گیت‌ها
نیاز دارد.

MLflow فقط config، source snapshot، receipt anchor، plan، sealها، metricها و
گزارش‌ها را دریافت می‌کند. صوت، embedding، checkpoint، optimizer و credential
روی سرور می‌مانند. انتقال به سیستم محلی فقط پس از promotion واقعی مجاز است.
