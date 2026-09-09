# صف پژوهش مدت‌کوتاه برای CAM++

## وضعیت و هدف

این سند صف پژوهش پس از F005 را تعریف می‌کند. اعداد زیر نتایج قطعی موجودند، نه برآورد مقاله‌ای:

| مورد | مقدار |
|---|---:|
| بهترین OOF فعلی، C002b/P002 | `0.9565282892229405` Macro-F1 |
| accuracy همان OOF | `0.9589313314197394` |
| نتیجهٔ ثبت‌شدهٔ P002 روی leaderboard | `0.9640736914724107` Macro-F1 |
| هدف پروژه | OOF گروه‌جدا و بازتولیدپذیر `>= 0.965` روی ۴۵۲۹ فایل و ۴۴۷ کلاس |
| فاصلهٔ OOF تا هدف | `0.0084717107770595` |

تشخیص فعلی این است که calibration به‌تنهایی کافی نیست. از ۲۲۱۷ فایل known و nonzero، هویت واقعی در ۲۱۴۰ فایل rank-1 است. در knownهای ردشده فقط ۱۲ فایل rank-1 درست دارند و ۶۵ فایل هویت واقعی‌شان پایین‌تر از rank-1 است. بنابراین بخشی مهم از خطاها به representation و ranking مربوط است.

مسئله در فایل کوتاه بسیار متمرکز است. در OOF فعلی، ۲۶ فایل known و nonzero کوتاه‌تر از ۳ ثانیه از ۱۹ speaker وجود دارد؛ ۲۲ خطا به ۱۸ speaker تعلق دارد. در عین حال، فایل known با سیگنال صفر با هیچ روش صوتی قابل بازیابی نیست و نباید از روی نام، گروه محتوا یا truth حدس زده شود.

مقادیر «سود مورد انتظار» در کارت‌های زیر استنباط مهندسی از نتایج مقالات و خطاهای این داده‌اند و تضمین نتیجه نیستند. EER مقاله مستقیماً معادل Macro-F1 مسابقه نیست.

## قرارداد مشترک همهٔ شاخه‌ها

1. هر arm فقط روی تمام ردیف‌های known با نقش `encoder_fit_allowed` آموزش می‌بیند. singletonها حذف یا به split داخلی مصنوعی منتقل نمی‌شوند.
2. انتخاب arm فقط با queryهای known و disjoint در calibration اصلی انجام می‌شود. unknown calibration تا seal شدن arm پنهان می‌ماند و سپس فقط برای برازش policy باز-مجموعه استفاده می‌شود.
3. outer truth تا seal شدن checkpoint، policy و تمام انتخاب‌ها خوانده نمی‌شود و هر outer فقط یک بار ارزیابی می‌شود.
4. برای انتساب علّی، candidate باید با fresh same-protocol control مقایسه شود. C002b با Macro-F1 برابر `0.9565282892229405` نیز comparator ثابت بهترین مدل واقعی باقی می‌ماند؛ ادعای بازتولید دقیق C002b با protocol تازه مجاز نیست.
5. بوت‌استرپ، paired و روی کل content group انجام می‌شود؛ mixed-label group تجزیه نمی‌شود و purity کلاس فرض نمی‌شود.
6. gate نهایی promotion برای همهٔ شاخه‌ها سخت و یکسان است: حداقل `+0.003` Macro-F1 نسبت به fresh control و C002b، افت هر fold حداکثر `0.001`، accuracy غیرکاهشی، short-known top-1 غیرکاهشی، بدون افزایش K→K، افزایش U→K حداکثر ۲، و lower bound بوت‌استرپ گروهی حداقل `-0.0005`.
7. تنها مدلی که promotion gate را کامل رد کند می‌تواند برای بسته‌بندی منتقل شود. checkpointهای ناموفق و cacheها server-only می‌مانند.

## کارت F006: Frozen Long Anchor

**فرضیه.** هدف بلندِ ثابت، نسبت به long embedding پویای F005، سیگنال کم‌نوسان‌تری برای short branch می‌دهد و از حرکت مرزهای speaker هنگام fine-tuning روی دادهٔ کوچک جلوگیری می‌کند.

**تغییر واحد.** crop plan، batch، head، AAM، optimizer، schedule و ضرایب arm برندهٔ F005 بدون تغییر می‌مانند. فقط `dynamic_same_step_stop_gradient` با embedding بلند تولیدشده توسط یک کپی منجمد از checkpoint پیش از tail جایگزین می‌شود. student long همچنان AAM می‌گیرد؛ frozen anchor فقط target consistency است. برای برابر ماندن هزینهٔ tail، anchorهای crop plan ثابت پیشاپیش cache می‌شوند.

**کنترل.** همان arm برندهٔ F005 با long target پویای stop-gradient و همان checkpoint آغازین.

**پشتوانه.** teacher-student برای short utterance از teacher بلند و منجمد و student کوتاه استفاده می‌کند و loss شباهت embedding را مؤثرتر از KLD تنها گزارش کرده است. روش Fixed Anchor جدید نیز نشان می‌دهد target ثابت می‌تواند از نوسان و محوشدن مرزها در joint optimization جلوگیری کند. انتقال نتیجهٔ Fixed Anchor از robustness نویز به robustness مدت یک استنباط آزمایشی است و باید با کنترل F005 اثبات شود.

**سود مورد انتظار.** `+0.001` تا `+0.003` Macro-F1، عمدتاً با کاهش خطای rank در knownهای کوتاه.

**ریسک.** anchor منجمد ممکن است mismatch دامنهٔ checkpoint عمومی را حفظ کند و adaptation مفید را محدود سازد.

**بودجه.** سقف `2 GPU-hour` برای دو outer fold، پس از microbenchmark واقعی F005.

**Go/no-go.** ورود به outer فقط وقتی مجاز است که known calibration از کنترل بهتر شود و short-known top-1 افت نکند. بهبود loss یا short/long cosine بدون بهبود queryهای held-out، no-go است. promotion نهایی باید تمام gateهای قرارداد مشترک را رد کند.

## کارت F007: L2-SP روی encoder

**فرضیه.** جریمهٔ فاصله از وزن‌های pretrained، در دادهٔ بسیار کم‌نمونه از weight decay معمول بهتر جلوی catastrophic drift را می‌گیرد، بدون اینکه head جدید کلاس‌های مسابقه را به classifier قدیمی ببندد.

**تغییر واحد.** regularization لایه‌های trainable encoder از L2 نسبت به صفر به

`L_SP = lambda / 2 * ||theta - theta_0||^2`

تغییر می‌کند. `theta_0` snapshot بایتی همان encoder در checkpoint پیش از tail است. head جدید همچنان regularization و schedule کنترل را دارد و به `theta_0` متصل نمی‌شود. دو مقدار از پیش ثبت‌شدهٔ `lambda in {0.01, 0.1}` بررسی می‌شوند؛ `0.1` مقدار گزارش‌شده در مطالعهٔ منبع است و `0.01` arm محافظه‌کارانهٔ اختلاف مقیاس معماری است.

**کنترل.** arm دقیق F005 با weight decay فعلی، crop plan و تعداد update یکسان.

**پشتوانه.** در یک corpus طبیعی و کوچک short-utterance، L2-SP در همهٔ انتخاب‌های لایه از L2 معمول بهتر بود و بهترین EER را از `10.52` به `9.85` رساند. آن داده ۳۷۵۵ utterance از ۲۲۸ speaker و حداقل سه نمونه برای هر speaker داشت؛ در نتیجه انتقال اندازهٔ سود به دادهٔ حاضر معتبر نیست، ولی جهت regularization با وضعیت singletonها سازگار است.

**سود مورد انتظار.** `+0.0005` تا `+0.0025` Macro-F1. ارزش اصلی، جلوگیری از regression اواخر tail است.

**ریسک.** lambda بزرگ encoder را عملاً منجمد می‌کند و lambda کوچک بی‌اثر می‌ماند. نسبت gradient جریمه به task loss باید ثبت شود و arm ناپایدار یا غالب‌شده زود متوقف شود.

**بودجه.** سقف `3 GPU-hour` برای دو lambda و دو outer fold، با reuse کامل control.

**Go/no-go.** این شاخه فقط وقتی اولویت دارد که F005 در checkpoint زودتر روی known calibration بهتر باشد و در ادامه regress کند یا drift قابل‌توجه نشان دهد. treatment باید هم drift را کاهش دهد و هم query metric را حفظ یا بهتر کند؛ بهبود train loss به‌تنهایی no-go است. promotion نهایی تابع gate مشترک است.

## کارت F008: DAME-FT-HW-Lite برای CAM++ 192D

**فرضیه.** واداشتن prefixهای ابتدایی CAM++ به تمرکز بر سرنخ‌های speaker در مدت کوتاه، ranking فایل‌های کوتاه را بهتر می‌کند، در حالی که embedding کامل ۱۹۲بعدی کیفیت فایل بلند را حفظ می‌کند.

**نسخهٔ منبع.** DAME-FT-HW برای encoderهای ۱۹۲بعدی از prefixهای `{48, 96, 192}`، مدت‌های `{1, 2, 6}` ثانیه و marginهای `{0.0, 0.2, 0.5}` استفاده می‌کند. inference همیشه embedding کامل ۱۹۲بعدی است. روی ECAPA-TDNN در VoxCeleb1-O، s-avg از `2.54` به `2.27` و 5s-1s از `4.84` به `4.39` EER رسید، اما نتیجه در همهٔ test setها و شرایط یکنواخت نبود.

**چرا بازتولید دقیق نامعتبر است.** مقاله chunkهای هر مدت را از utteranceهای متفاوت همان speaker می‌گیرد، روی VoxCeleb2-dev با batch شامل ۱۲۸ speaker و ۳۰ epoch آموزش می‌دهد و در fine-tuning از classifier همان speakerهای pretraining بهره می‌برد. دادهٔ هر outer این پروژه فقط ۶۶۲/۶۶۷ ردیف برای ۴۴۶ label دارد و singleton فراوان است. بنابراین آزمایش باید صریحاً `DAME-lite` نام بگیرد و ادعای replication مقاله نکند.

**تغییر واحد.** پس از head warmup مشترک، control برای هر سه crop از کل ۱۹۲ بعد استفاده می‌کند: `{192, 192, 192}`. treatment فقط نگاشت duration-to-prefix را به `{1s->48, 2s->96, 6s->192}` تغییر می‌دهد. cropها، marginها، loss weight، تعداد forward، optimizer، schedule و checkpoint آغازین در دو arm یکسان‌اند. برای singleton، cropها nested و از همان waveform هستند.

**کنترل.** D-ALMFT-like با سه مدت و سه margin یکسان، اما supervision کل ۱۹۲ بعد در هر سه مدت.

**سود مورد انتظار.** `+0.001` تا `+0.004` Macro-F1؛ بیشترین ظرفیت بالقوه در این صف، همراه با بیشترین عدم‌قطعیت.

**ریسک.** fresh head و singletonها ممکن است prefixهای ابتدایی را به فایل حفظ کنند. وابستگی ابعاد ابتدایی به ترتیب تصادفی embedding ممکن است با CAM++ مانند ECAPA رفتار نکند. DAME-FT-SW به‌دلیل افت مکرر full-duration وارد صف نمی‌شود.

**بودجه.** سقف `6 GPU-hour` برای control و treatment روی دو outer fold.

**Go/no-go.** علاوه بر gate مشترک، short-known top-1 باید بهتر و full-duration known accuracy غیرکاهشی باشد. prefixهای ۴۸ و ۹۶ در inference فقط diagnostic هستند؛ انتخاب adaptive prefix با outer truth ممنوع است. اگر control مدت‌متغیر به‌اندازهٔ treatment بهتر شود، اثر به DAME نسبت داده نمی‌شود.

## Temporal replication padding: فقط preflight، نه شاخهٔ اصلی

در frontend فعلی، waveform تنها وقتی کوتاه‌تر از یک ثانیه است zero-pad می‌شود؛ در این ناحیه فقط سه فایل known و nonzero وجود دارد و هر سه اکنون اشتباه‌اند. repeat-to-3s دامنه را به ۲۶ فایل known و nonzero از ۱۹ speaker گسترش می‌دهد که ۲۲ خطا دارند، اما هم‌زمان ۸۷ فایل unknown کوتاه‌تر از سه ثانیه را نیز تغییر می‌دهد؛ false confidence و U→K ریسک واقعی‌اند.

شاهد peer-reviewed موجود برای repeat padding به RNN متن‌وابسته مربوط است. آن مطالعه می‌گوید تکرار frame اطلاعات speaker را حفظ می‌کند ولی ترتیب phrase را تغییر می‌دهد؛ روش پیشنهادی و برندهٔ مقاله حذف خروجی‌های padded است، نه اثبات برتری repetition برای CAM++ متن‌مستقل. CD-VAT نیز replication را فقط برای ورودی کوتاه‌تر از پنجره به کار برده و ablation مستقل برای سود آن ارائه نکرده است. تکرار waveform اطلاعات آوایی تازه نمی‌سازد.

در صورت نیاز فقط یک preflight با سقف `0.25 GPU-hour` روی known calibration مجاز است: frontend فعلی در برابر circular-repeat-to-3s، با encoder و policy ثابت. اگر دست‌کم دو speaker مجزا اصلاح نشوند، آزمایش بدون outer evaluation بسته می‌شود. این preflight به‌خودی‌خود مجوز تغییر frontend بستهٔ نهایی نیست.

## مسیر اجرای صف

```text
F005: dynamic long-short consistency
  |
  +-- short/long alignment بهتر، ولی target drift یا Macro-F1 plateau دارد -> F006
  |
  +-- checkpoint زودهنگام بهتر است و tail بعداً regress می‌کند          -> F007
  |
  +-- consistency سیگنال پایدار ندارد یا مسیر anchor کافی نیست          -> F008
```

ترتیب عملی پایه `F005 -> F006 -> F007 -> F008` است، اما triggerهای بالا اجازه می‌دهند شاخهٔ فاقد پیش‌شرط رد شود و GPU مصرف نشود. ترکیب روش‌ها در این سه اجرا ممنوع است. فقط اگر دو مؤلفه جداگانه و با کنترل علّی gate خود را رد کنند، یک آزمایش ترکیبی جدید و از پیش ثبت‌شده می‌تواند آن‌ها را ترکیب کند.

## بک‌لاگ سطح دوم پس از مرور SciSpace

جست‌وجوی هدفمند SciSpace در ۹ سپتامبر ۲۰۲۶ دو خانوادهٔ دیگر را پیدا کرد، اما شواهد فعلی برای جلو انداختن آن‌ها از F005 تا F008 کافی نیست:

- **F009، segment/crop aggregation:** مقالهٔ Crop Aggregating، embedding چند قطعه و embedding تجمیعی را هم‌زمان آموزش می‌دهد و برای آزمون یک‌ثانیه‌ای بهبود نسبی بزرگی گزارش می‌کند. این روش برای فایل‌هایی که چند قطعهٔ مستقل آوایی دارند منطقی است؛ در دادهٔ ما سخت‌ترین جمعیت خود فایل‌های زیر سه ثانیه‌اند و تقسیم آن‌ها شواهد تازه تولید نمی‌کند. این شاخه فقط اگر تحلیل F005 نشان دهد خطاهای باقی‌مانده در فایل‌های بلند با ناپایداری crop مرتبط‌اند، با کنترل «full-utterance inference» وارد preflight می‌شود.
- **F010، imbalance-length episodic training:** روش meta-learning با support بلند، query کوتاه و طبقه‌بندی روی همهٔ کلاس‌های train طراحی شده است. بخش طبقه‌بندی سراسری با هدف Macro-F1 ما سازگار است، اما singletonهای فراوان باعث می‌شوند support و query اغلب cropهای همان فایل باشند و خطر یادگیری خصوصیات ضبط بالا برود. این شاخه تنها پس از شکست روش‌های ساده‌تر و با group-disjoint validation مستقل ارزش بودجهٔ GPU دارد.

این دو مورد فعلاً `deferred` هستند. وجود مقاله به‌تنهایی آن‌ها را به آزمایش مجاز تبدیل نمی‌کند؛ trigger داده‌ای، کنترل paired و بودجه باید پیش از اجرا ثبت شوند.

## روش‌های ردشده یا پرریسک

- **Triplet، GE2E، supervised contrastive و centroid adaptation با positive بین‌utterance:** singletonها positive یا centroid مستقل فراهم نمی‌کنند. cropهای یک فایل نباید به‌عنوان session مستقل معرفی شوند.
- **Centroid Alignment مقالهٔ LVC:** در ۱ و ۲ ثانیه امیدوارکننده بود، اما centroid پایدار چند utterance برای بیشتر speakerهای fit موجود نیست.
- **DAME دقیق و DAME-FT-SW:** فرض داده و classifier نسخهٔ دقیق برقرار نیست و SW در fine-tuning بارها full-duration را خراب کرده است.
- **Full-encoder LMFT با margin بزرگ:** هم خطر overfit روی ۶۶۲/۶۶۷ ردیف دارد و هم LMFT استاندارد در نتایج DAME غالباً short-duration را بدتر کرده است.
- **Temporal replication به‌عنوان راه ساخت اطلاعات:** repetition تنها padding را عوض می‌کند و zero-signal یا phonetic evidence مفقود را بازسازی نمی‌کند.
- **حدس label برای zero-signal:** هر بهبود ظاهری از filename، content-group یا outer truth نشانهٔ memorization یا leakage است.
- **Pseudo-label از outer/test، tuning روی outer یا شکستن content group:** با قرارداد ارزیابی مسابقه ناسازگار است.
- **Calibration/QMF بیشتر به‌عنوان راه اصلی:** oracle و rank analysis نشان داده‌اند ranking خطا دارد؛ threshold نمی‌تواند هویت درست را از rank پایین به rank اول منتقل کند.
- **افزودن هم‌زمان augmentation، loss، padding و layer scope:** attribution را از بین می‌برد و روی development OOF بارها دیده‌شده، انتخاب خوش‌بینانه می‌سازد.

## منابع اولیه

- [DAME: Duration-Aware Matryoshka Embedding for Duration-Robust Speaker Verification، ICASSP 2026](https://arxiv.org/html/2601.13999)
- [Length- and Noise-Aware Training Techniques for Short-Utterance Speaker Recognition؛ LVC و Centroid Alignment](https://www.isca-archive.org/interspeech_2020/chen20p_interspeech.pdf)
- [Short Utterance Compensation via Cosine-Based Teacher-Student Learning](https://arxiv.org/abs/1810.10884)
- [Open-Set Short Utterance Forensic Speaker Verification؛ teacher-student و L2-SP](https://www.isca-archive.org/interspeech_2020/sang20_interspeech.pdf)
- [A Stage-Wise Learning Strategy with Fixed Anchors for Robust Speaker Verification، ICASSP 2026](https://arxiv.org/pdf/2510.18530)
- [Crop Aggregating for Short Utterances Speaker Verification Using Raw Waveforms](https://arxiv.org/abs/2005.03329)
- [Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs](https://arxiv.org/abs/2004.02863)
- [A Simple Distortion-Free Method to Handle Variable-Length Sequences؛ padding ablation](https://www.mdpi.com/2076-3417/10/12/4092)
- [Cosine-Distance Virtual Adversarial Training؛ کاربرد محدود replication padding](https://www.interspeech2020.org/uploadfile/pdf/Wed-3-5-7.pdf)
- [CAM++: A Fast and Efficient Network for Speaker Verification](https://www.isca-archive.org/interspeech_2023/wang23ha_interspeech.html)
- [نسخهٔ رسمی recipeهای CAM++ در 3D-Speaker](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/egs/voxceleb/sv-cam%2B%2B/conf/cam%2B%2B.yaml)
