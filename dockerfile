# استخدم صورة بايثون الرسمية
FROM python:3.11-slim

# تثبيت المكتبات اللازمة لـ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات المتطلبات أولاً لتسريع عملية البناء
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# تعريف المنافذ
EXPOSE 8005  
# للصور

# يمكنك اختيار أي API تريد تشغيله بشكل افتراضي
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8006"]

