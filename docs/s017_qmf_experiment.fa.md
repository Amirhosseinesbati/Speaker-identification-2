# S017: کالیبراسیون QMF برای تصمیم open-set روی C002b

## هدف و منبع ثابت

S017 یک آزمایش پس‌پردازش کم‌پارامتر روی embeddingهای ثابت `C002b` است. هیچ
صوتی decode نمی‌شود، هیچ encoderی forward یا به‌روزرسانی نمی‌شود و cache
تازه‌ای ساخته نمی‌شود. تنها منبع مجاز، مسیر زیر روی سرور است:

```text
/workspace/Speaker-identification-2-c002/artifacts/training/cuda_gain_c002/C002_20260908T230219Z_479a211696d64a9495854f29006f72f7
```

هویت منبع با parent run برابر `37589a011a8c4f9aa0c5de0936ac7e46`، child
`C002b` برابر `14a0f144b27a46e7a36e3fc3b2ce24c4` و commit برابر
`adf820c253c985364ccc36b24d2943508a661fc4` قفل شده است. پیش از هر برازش،
SHA-256 گزارش اصلی، گزارش اجرای جفتی، manifest و identity مربوط به cache با
مقادیر config مقایسه می‌شوند. cache فقط‌خواندنی است و شکست هر کنترل، عملیات
را پیش از ایجاد MLflow run متوقف می‌کند.

## فرضیه و نامزدهای بسته

مدل پایه همان ranking شناسه‌های known در C002b را حفظ می‌کند. فرضیه این است
که logistic regressor کوچک بتواند با شکل scoreها، و در نسخهٔ دوم با افزودن
کیفیت صوت، تصمیم known/unknown را بهتر کند. target دقیقاً `is_known` است؛ پس
QMF هیچ UUID شناخته‌شده‌ای را با UUID دیگر جابه‌جا نمی‌کند.

فقط سه سیاست وجود دارد:

1. `baseline`: تصمیم C002b بدون تغییر؛
2. `qmf_scores`: مدل QMF با `scores_only`؛
3. `qmf_scores_quality`: همان مدل با `scores_quality`.

`scores_only` دقیقاً شامل این ۹ ویژگی است:

```text
fused_known_top
fused_known_gap
fused_unknown_top
fused_unknown_top3_mean
fused_unknown_top50_mean
fused_unknown_top50_std
public_known_minus_unknown
advanced_known_minus_unknown
encoder_winner_agreement
```

`scores_quality` همان ۹ ویژگی را با
`log1p_duration_capped180` و `rms_dbfs_clipped` تکمیل می‌کند. هیچ feature-set،
penalty یا مدل دیگری جست‌وجو نمی‌شود. penalty مستقیم L2 برابر `0.1`، scale
floor برابر `1e-6` و تعداد quantileهای threshold دقیقاً ۲۰۱ است. وزن هر
content-group ابتدا برابر و سپس دو target known/unknown متوازن می‌شود. مدل
شامل ترتیب ویژگی‌ها، mean، scale، coefficientها و intercept در JSON محدود و
finite صادر می‌شود؛ inference آن فقط به NumPy نیاز دارد و pickle/joblib ندارد.

## جداسازی کامل و انتخاب داخلی

S017 عمداً همان تقسیم‌های سه‌گانهٔ تثبیت‌شدهٔ S012 و salt برابر
`S012-nested-gallery-v1` را دوباره استفاده می‌کند. این کار تقسیم تازه‌ای پس از
مشاهدهٔ نتایج قبلی نمی‌سازد و هویت هر content-group را ثابت نگه می‌دارد.

در هر outer fold، گروه اعتبارسنجی هر meta-fold هم‌زمان از برازش logistic،
محاسبهٔ scaler، gallery گویندگان شناخته‌شده، cohort ناشناخته، کالیبراسیون پایه
و انتخاب threshold کنار گذاشته می‌شود. مدل logistic و threshold همان case هر دو
فقط با ردیف‌های `case.fit` ساخته می‌شوند و سپس دقیقاً یک‌بار روی
`case.validation` مجزا اعمال می‌شوند. score و ویژگی ردیف اعتبارسنجی نیز با
مراجع مجاز همان meta-fold دوباره ساخته می‌شوند؛ cross-fit کردن صرف ضرایب
logistic یا انتخاب threshold روی همهٔ meta-OOFها معتبر نیست.

پیش‌بینی‌های held-out سه meta-fold تجمیع می‌شوند و فقط همان‌ها مبنای انتخاب
recipe هستند. پس از این ارزیابی، logistic کامل و threshold کامل هر دو با
ردیف‌های full-inner ساخته می‌شوند و روی outer اعمال می‌شوند. QMF تنها وقتی از fallback پایه عبور
می‌کند که رشد Macro-F1 تجمیعی meta حداقل `0.001` باشد و افت هیچ meta-fold بیش
از `0.002` نشود. در غیر این صورت `baseline` انتخاب می‌شود. ترتیب شکستن تساوی
ثابت است: baseline، سپس scores-only و بعد scores-quality. سیاست و مدل نهایی
پیش از دسترسی به برچسب outer مهر می‌شوند؛ سپس روی دادهٔ مجاز outer-training
بازبرازش و هر outer fold فقط یک بار ارزیابی می‌شود.

## گیت سخت ارتقای بیرونی

نامزد فقط با برآورده‌شدن هم‌زمان همهٔ شروط زیر اجازهٔ ساخت release candidate
دارد:

- افزایش Macro-F1 تجمیعی حداقل `0.0015`؛
- تغییر Macro-F1 هر دو outer fold حداقل صفر؛
- تغییر accuracy تجمیعی حداقل صفر؛
- افزایش unknown→known حداکثر دو فایل؛
- افزایش known→other-known دقیقاً صفر؛
- کران پایین bootstrap جفتی حداقل `-0.001`؛
- برابری دقیق تصمیم مسیر CPU و CUDA.

شکست یک شرط، C002b را بدون تغییر حفظ می‌کند. عبور از گیت فقط ساخت و QA یک
کاندید را مجاز می‌کند؛ OOF توسعه قبلاً بارها بررسی شده و آزمون مستقل یا برآورد
تضمینی لیدربورد نیست.

ممکن است selector در دو outer fold، QMF یا threshold متفاوتی انتخاب کند. این
رفتار برای ارزیابی OOF یک فرایند nested معتبر است، اما مدل‌های همان foldها
مستقیماً مدل نهایی لیدربورد نیستند. اگر گیت عبور کند، مرحلهٔ بسته‌بندی باید
انتخاب nested را روی کل training تکرار و دقیقاً یک مدل نهایی را refit کند.

threshold هر fit روی logits درون‌آموزشی همان logistic انتخاب می‌شود. این تصمیم
به‌صورت یک procedure ثابت بین meta و outer آینه می‌شود و گروه held-out را وارد
fit نمی‌کند؛ بااین‌حال خود threshold می‌تواند روی fit overfit شود و به همین دلیل
فقط عملکرد held-out meta و outer برای قضاوت استفاده می‌شود.
bootstrap جفتی نیز به ۴۴۷ کلاس مشاهده‌شده شرطی است و فقط سنجهٔ پایداری مکمل
است، نه فاصلهٔ اطمینان کامل برای توزیع پنهان.

## اجرا، ثبت و نگهداری

launcher پیش‌فرض فقط قرارداد را اعتبارسنجی می‌کند:

```bash
python scripts/score_qmf.py
```

اجرای واقعی به پرچم صریح و binding معتبر experiment شمارهٔ ۱ نیاز دارد:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/score_qmf.py --execute
```

MLflow فقط config حل‌شده، هش‌های منبع، snapshot کد `src`، مدل‌های JSON،
metricها، گزارش‌ها، پیش‌بینی‌ها و رسید bootstrap را دریافت می‌کند. صوت خام،
embedding، وزن encoder و credential هرگز بارگذاری نمی‌شوند.

بدون عبور کامل از گیت ارتقا، هیچ خروجی به سیستم کاربر منتقل نمی‌شود و cache،
مدل‌های برازش‌شده، آرایه‌های میانی و کل run directory روی سرور می‌مانند. پس
از عبور گیت نیز انتقال به config حل‌شده، رسید هش/تأیید و بستهٔ حداقلی آفلاین
لیدربورد محدود است.
