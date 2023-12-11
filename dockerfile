FROM python:3.8


WORKDIR C:\Users\WebUser\Desktop\М.Тех_ТЗ_DS\М.Тех_ТЗ_DS

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]