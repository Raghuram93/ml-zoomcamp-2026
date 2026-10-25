FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

Install production server and scikit-learn dependency

RUN pip install --no-cache-dir 

--trusted-host pypi.org 

--trusted-host files.pythonhosted.org 

--trusted-host pypi.python.org 

flask gunicorn scikit-learn==1.6.1

Copy the v2 predict script into the container

COPY predict_v2.py .

EXPOSE 9696
Serve predict_v2 using Gunicorn WSGI server

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict_v2:app"]
