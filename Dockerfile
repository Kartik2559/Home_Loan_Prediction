FROM python
WORKDIR /homeapp
COPY . /homeapp
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run home_loan.py
