import os
import torch
from flask import Flask, request, jsonify, render_template
# import joblib
import transformers
import sqlite3
from flask_cors import CORS
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from os.path import dirname
from gtts import gTTS
from googletrans import Translator
import gc         
from accelerate import init_empty_weights,Accelerator
import json
import time
from langchain.llms import CTransformers
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import numpy as np


is_ivr = False

app = Flask(__name__)
CORS(app)


class NewTokenHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens_generated = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running."""
        self.num_tokens_generated = 0
        self.start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        total_time = time.time() - self.start_time
        print(f"\n\n {self.num_tokens_generated} tokens generated in {total_time:.2f} seconds")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.num_tokens_generated += 1
        print(f"{token}", end="", flush=True)

def load_sms_model():
    sms = CTransformers(
        model='quantizedmodels/ggml-sms-model-q8_0.bin', # Location of downloaded GGML model
        model_type='llama', # Model type Llama
        stream=False,
        callbacks=[NewTokenHandler()],
        config={
            'max_new_tokens': 512,
            'temperature': 0.01,
            'stop': "<0x0A>"
        }
    )
    return sms

def load_ivr_model():
    ivr = CTransformers(
        model='quantizedmodels/ggml-ivr-model-fp16.bin', # Location of downloaded GGML model
        model_type='llama', # Model type Llama
        stream=False,
        callbacks=[NewTokenHandler()],
        config={
            'max_new_tokens': 512,
            'temperature': 0.01,
            'stop': "<0x0A>"
        }
    )
    return ivr

if is_ivr:
    ivr_model = load_ivr_model()
else:
    sms_model = load_sms_model()

# helper functions

def generate_ivr(hedis_measure, response_type, prompt):
    prompt = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=ivr_model, prompt=prompt)

    result = chain({'query': prompt}, return_only_outputs=True)

    return result

def generate_sms(hedis_measure, from_age, to_age, response_type, prompt):
    prompt = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=sms_model, prompt=prompt)

    result = chain({'query': prompt}, return_only_outputs=True)

    return result

def generate_email(hedis_measure, from_age, to_age, response_type, prompt):
    prompt = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=sms_model, prompt=prompt)

    result = chain({'query': prompt}, return_only_outputs=True)

    return result


DATABASE = 'database.db'

def connect_db():
    return sqlite3.connect(DATABASE)

# # API endpoint to get data from the Bucket table
# @app.route('/api/bucket', methods=['GET'])
# def get_bucket_data():
#     try:
#         conn = connect_db()
#         cursor = conn.cursor()

#         # Assuming the Bucket table has a 'name' column
#         cursor.execute('SELECT name FROM Bucket')
#         data = cursor.fetchall()

#         # Convert data to a list of names
#         bucket_names = [row[0] for row in data]

#         conn.close()

#         return jsonify({'bucket_names': bucket_names})

#     except Exception as e:
#         return jsonify({'error': str(e)})
    
# API endpoint to get data from the HedisMeasure table
@app.route('/api/hedis_measure', methods=['GET'])
def get_hedis_measure_data():
    try:
        conn = connect_db()
        cursor = conn.cursor()

        # Assuming the Bucket table has a 'name' column
        cursor.execute('SELECT name FROM HedisMeasure')
        data = cursor.fetchall()

        # Convert data to a list of names
        hedis_measure_names = [row[0] for row in data]

        conn.close()

        return jsonify({'hedis_measure_names': hedis_measure_names})

    except Exception as e:
        return jsonify({'error': str(e)})

# # Endpoint to create a new guide
# @app.route('/generate', methods=["POST"])
# def generate():
#     hedis_measure = request.json['hedis_measure']
#     bucket = request.json['bucket']
#     generate_type = request.json['type']

#     generate_type_str = str(generate_type)


#     response_data = {
#                 'hedis_measure': hedis_measure,
#                 'bucket': bucket,
#                 'generate_type': generate_type_str
#             }

#     return jsonify(response_data) 



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate',  methods=['GET'])
def generate():
    if request.method == 'GET':
        hedis_measure = request.args.get('hedis')
        from_age = request.args.get('from')
        to_age = request.args.get('to')
        response_type = request.args.get('type')

        if response_type == 'sms':
            prompt = f'[INST] Generate one sms informing customer about {hedis_measure}. Age group: {from_age}-{to_age} [/INST] '
            data = generate_sms(hedis_measure, from_age, to_age, response_type, prompt)
            data = [{
                'data': data,
                'data_array': []
            }]

        elif response_type == 'email':
            prompt = f'[INST] Generate an email for informing customer about {hedis_measure}. Age group: {from_age}-{to_age} [/INST]'
            data = generate_email(hedis_measure, from_age, to_age, response_type, prompt)
            data = [{
                'data': data,
                'data_array': []
            }]

        else:
            prompt = f'Please generate five questions for the customer having the following \n\n Hedis Measure:\n{hedis_measure}\n'
            data = generate_ivr(hedis_measure, response_type, prompt)
            res = data.values()
            d = list(res)
            res_arr = np.array(d)
            # print(res_arr)
            x = res_arr[0].split('\n')
            data_array = [i for i in x if i != '']

            data = [{
                'data': data,
                'data_array': data_array
            }]

    return jsonify(data) 

@app.route('/reset-memory')
def reset():
    pass

@app.route('/translation', methods=['GET'])
def translate():
    if request.method == 'GET':
        text_to_translate = request.args.get('text')
        # Create a Translator object
        translator = Translator()
        # Translate the text to Spanish
        translated_text = translator.translate(text_to_translate, src='en', dest='es')
        prepared_response = { 
            "type" : 'txt', 
            "data" : translated_text.text, 
        } 
    return jsonify(prepared_response)

@app.route('/generate-audio', methods=['GET'])
def genereate_audio():
    if request.method == 'GET':
        text_to_convert = request.args.get('text')
        translate_to = request.args.get('lang')
       
        # spanish
        if translate_to == 'es':
            language = 'es'
            # Create a gTTS object
            tts = gTTS(text=text_to_convert, lang=language, slow=False)
            # Save the audio file
            tts.save('output_audio_spanish.mp3')
            saved_path = f'{dirname(__file__)}\\output_audio_spanish.mp3'
        else:
            language = 'en'
            # Create a gTTS object
            tts = gTTS(text=text_to_convert, lang=language, slow=False)
            # Save the audio file
            tts.save('output_audio_english.mp3')
            saved_path = f'{dirname(__file__)}\\output_audio_english.mp3'

        prepared_response = { 
            "type" : 'message', 
            "data" : f'Audio save at location: {saved_path}', 
        } 

    return jsonify(prepared_response)


if __name__ == '__main__':
    app.run(port=5000)