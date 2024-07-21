import json
import requests
import ast, os
import time
import numpy as np
import pandas as pd
import torch

from transformers import PatchTSTForRegression
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

model = PatchTSTForRegression.from_pretrained("namctin/patchtst_etth1_regression")

#=========================Creds related to watsonx.ai===============================
WX_API_KEY = os.environ['WX_API_KEY']
WX_PROJECT_ID= os.environ['WX_PROJECT_ID']
WX_URL= os.environ['WX_URL']

creds = {
    "url": WX_URL,
    "apikey": WX_API_KEY 
}


#=========================function related to watsonx.ai===============================
def send_to_watsonxai(prompt, creds=creds, project_id=WX_PROJECT_ID,
                    model_name='meta-llama/llama-3-70b-instruct', #'mistralai/mixtral-8x7b-instruct-v01',', #'meta-llama/llama-2-13b-chat', #
                    decoding_method="greedy",
                    max_new_tokens=100,
                    min_new_tokens=1,
                    temperature=0,
                    repetition_penalty=1.0,
                    stop_sequences=[],
                    ):
    '''
   helper function for sending prompts and params to Watsonx.ai
    
    Args:  
        prompts:list list of text prompts
        decoding:str Watsonx.ai parameter "sample" or "greedy"
        max_new_tok:int Watsonx.ai parameter for max new tokens/response returned
        temperature:float Watsonx.ai parameter for temperature (range 0>2)
        repetition_penalty:float Watsonx.ai parameter for repetition penalty (range 1.0 to 2.0)

    Returns: None
        prints response
    '''

    assert not any(map(lambda prompt: len(prompt) < 1, prompt)), "make sure none of the prompts in the inputs prompts are empty"

    # Instantiate parameters for text generation
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)
    
    
    output = model.generate_text(prompt)
    return output


def send_to_watsonxai_stream(prompt, creds=creds, project_id=WX_PROJECT_ID,
                    model_name= 'meta-llama/llama-3-8b-instruct',#'meta-llama/llama-3-70b-instruct', #'mistralai/mixtral-8x7b-instruct-v01',', #'meta-llama/llama-2-13b-chat', #
                    decoding_method="greedy",
                    max_new_tokens=300,
                    min_new_tokens=1,
                    temperature=0,
                    repetition_penalty=1.0,
                    stop_sequences=["\n","\n\n"],
                    ):
    '''
   helper function for sending prompts and params to Watsonx.ai
    
    Args:  
        prompts:list list of text prompts
        decoding:str Watsonx.ai parameter "sample" or "greedy"
        max_new_tok:int Watsonx.ai parameter for max new tokens/response returned
        temperature:float Watsonx.ai parameter for temperature (range 0>2)
        repetition_penalty:float Watsonx.ai parameter for repetition penalty (range 1.0 to 2.0)

    Returns: None
        prints response
    '''

    assert not any(map(lambda prompt: len(prompt) < 1, prompt)), "make sure none of the prompts in the inputs prompts are empty"

    # Instantiate parameters for text generation
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)
    
    output = model.generate_text_stream(prompt)

    for chunk in output:
        yield chunk


#=========================function related to prediction===============================
def get_prediction(input_dict):
    predict_detail = input_dict['predict_detail']
    data_point_number = int(input_dict['data_point_number'])
    satuan = input_dict['satuan']

    if predict_detail == 'daily':
        multiplier = 100
    elif predict_detail == 'weekly':
        multiplier = 1000
    elif predict_detail == 'monthly':
        multiplier = 10000
    else:
        multiplier = 1

    if satuan== "Celcius":
        multiplier = 200
    
    past_values = torch.randint(low=1, high=3, size=(data_point_number, 512, 6))
    outputs = model(past_values=past_values)
    regression_outputs = outputs.regression_outputs

    data=abs(regression_outputs)
    avg = data.mean().item()
    data = data.tolist()

    data = [point[0] - avg if point[0] > avg else point[0] + avg for point in data]
    data = [point * multiplier for point in data]

    return data

#=========================function related to QNA===============================
def question_detail(user_question, model_name='meta-llama/llama-3-8b-instruct'):
    prompt =f"""Tugas anda adalah memahami pertanyaan yang diberikan oleh pengguna.
    Dari pertanyaan tersebut, tarik kesimpulan apakah yang diinginkan adalah hourly, daily, atau monthly.
    Pahami juga berapa jumlah data pointnya.
    Cari informasi mengenai satuan yang tepat untuk datapoint.

    Berikut adalah contohnya jika angka di sebutkan secara spesifik:
    Pertanyaan: saya minta prediksi status penjualan 3 hari ke depan!
    Jawaban: {{'predict_detail':'daily', 'data_point_number':'3', 'satuan':'Rupiah'}}

    Pertanyaan: hai, bisa tolong predict electricity load untuk minggu depan?
    Jawaban: {{'predict_detail':'daily', 'data_point_number':'7', 'satuan':'MW'}}

    Pertanyaan: forecast temperature buat 6 bulan ke depan!
    Jawaban: {{'predict_detail':'monthly', 'data_point_number':'6', 'satuan':'Celcius'}}

    Berikut adalah contohnya jika angka tidak di sebutkan secara spesifik:
    Pertanyaan: besok oil temperaturenya berapa?
    Jawaban: {{'predict_detail':'hourly', 'data_point_number':'24', 'satuan':'Celcius'}}

    Pertanyaan: saya minta anomaly scores untuk bulan depan!
    Jawaban: {{'predict_detail':'daily', 'data_point_number':'30', 'satuan':''}}

    Pertanyaan: tolong berikan saya analisis trend load electricity untuk tahun depan!
    Jawaban: {{'predict_detail':'monthly', 'data_point_number':'30', 'satuan':'MW'}}

    Pertanyaan: {user_question}
    Jawaban:"""

    wxai_output = send_to_watsonxai(prompt, stop_sequences=["\n\n"], model_name=model_name)
    wxai_output = ast.literal_eval(wxai_output.strip())

    predict_detail = wxai_output['predict_detail']
    satuan = wxai_output['satuan']

    data = get_prediction(wxai_output)
    output = []
    # getting desired output from tuple
    for count, value in enumerate(data):
        output.append({predict_detail:count+1 ,satuan:value})

    wxai_output['output'] = output
    wxai_output['data'] = data
    wxai_output['user_question'] = user_question
    wxai_output['visual'] = {"chart": "True", "x": predict_detail, "y": satuan}
    return wxai_output


def query_wxai(input, streaming=False,  model_name='meta-llama/llama-3-70b-instruct'):
    start_time = time.time()
    user_question = input['user_question']
    predict_detail = input['predict_detail']
    satuan = input['satuan']
    prediction_output =  input['data']

    prompt =f"""Tugas anda adalah menjawab pertanyaan yang diberikan oleh pengguna.
    Pertanyaan user adalah {user_question}.
    Berikut adalah hasil prediksi  dari machine learning {prediction_output} dalam satuan {satuan} dengan detail prediksi {predict_detail}.
    Berikan jawaban yang sesuai, akurat, dan mudah dipahami oleh manusia.
    Hindari penggunaan karakter khusus, atau line baru jika tidak perlu.
    Hindari memberikan penjelasan mengenai cara pengerjaan, code, catatan dan sebagainya.
    Gunakan data point hanya dari prediksi machine learning.
    Jika ingin menampilkan data point, tampilkan desimal hanya hingga 2 angka di belakang titik.
    Jika data point terlalu banyak, gunakan range dan highlight data point yang penting saja.
    Berikan saran atau tindakan yang tepat berdasarkan hasil penilaian anda terhadap hasil prediksi machine learning.
    Jawaban:"""
        
    print("streaming =", streaming)

    if streaming:
        return send_to_watsonxai_stream(prompt, creds=creds, project_id=WX_PROJECT_ID, stop_sequences=['"""',"'''","```"], 
                                        model_name=model_name, max_new_tokens=400)

    else:
        result = send_to_watsonxai(prompt, creds=creds, project_id=WX_PROJECT_ID, stop_sequences=['"""',"'''","```"], 
                                        model_name=model_name, max_new_tokens=400)
        eta_wx = time.time() - start_time
        print("eta_wx: ", eta_wx)
        return result, eta_wx
    

