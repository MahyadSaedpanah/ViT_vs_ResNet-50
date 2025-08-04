vision_project/
│
├── data/
│   ├── prepare_data.py        # دانلود و پیش‌پردازش داده‌ها
│   └── augmentations.py       # اعمال افزایش داده
│
├── models/
│   ├── vit_model.py           # آماده‌سازی و تغییر ساختار ViT
│   └── resnet_model.py        # آماده‌سازی و تغییر ساختار ResNet
│
├── train/
│   ├── train_vit.py           # آموزش ViT
│   └── train_resnet.py        # آموزش ResNet
│
├── evaluate/
│   ├── evaluate.py            # محاسبه دقت، زمان، سایز و ...
│   └── robustness.py          # ارزیابی مقاومت با نویز (اختیاری)
│
├── visualize/
│   ├── vit_attention.py       # بصری‌سازی attention در ViT
│   └── resnet_gradcam.py      # Grad-CAM برای ResNet
│
├── utils/
│   └── helpers.py             # توابع کمکی مثل شمارش پارامتر، ذخیره مدل و ...
│
├── configs/
│   └── config.yaml            # تنظیمات قابل تغییر (batch size, lr و ...)
│
├── main.py                    # اجرای پروژه از ابتدا تا انتها
└── README.md
