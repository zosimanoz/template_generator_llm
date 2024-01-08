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


is_ivr = True

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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
)

def load_sms_model():
    model_name = f'{dirname(__file__)}\\smsmodel'
    peft_model_id = f'{dirname(__file__)}\\smsmodel'
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map='auto')
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, quantization_config=bnb_config, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    return model, tokenizer

# def load_ivr_model():
#     model_name = f'{dirname(__file__)}\\ivrmodel'
#     peft_model_id = f'{dirname(__file__)}\\ivrmodel'
#     config = PeftConfig.from_pretrained(peft_model_id)

#     # with init_empty_weights():
#     #     model_temp = AutoModelForCausalLM.from_config(config)

#     # device_map = infer_auto_device_map(model_temp)
#     # device_map["model.decoder.layers.37"] = "disk"

#     # base_model.model.model.norm, base_model.model.lm_head, base_model.model.model.layers.

#     model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16,device_map="auto",offload_folder='offload/',low_cpu_mem_usage=True, offload_state_dict = True)
#     model.config.use_cache = False
    
#     # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, quantization_config=bnb_config, device_map='auto')
#     tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     # Load the Lora model
#     model = PeftModel.from_pretrained(model, peft_model_id, offload_folder = "offload/")
#     model = model.merge_and_unload()

#     model.save_pretrained('final_ivr_model')

#     return model, tokenizer


def load_ivr_model():
    llm = CTransformers(
        model='quantizedmodels/ggml-ivr-model-fp16.bin', # Location of downloaded GGML model
        model_type='llama', # Model type Llama
        stream=True,
        callbacks=[NewTokenHandler()],
        config={
            'max_new_tokens': 256,
            'temperature': 0.01,
            'stop': "<0x0A>"
        }
    )
    return llm

if is_ivr:
    ivr_model = load_ivr_model()
else:
    sms_model, sms_tokenizer = load_sms_model()



# helper functions

# def generate_sms(hedis_measure, from_age, to_age, response_type, prompt):
#     # load model
#     # model_name = f'{dirname(__file__)}\\smsmodel'
#     # peft_model_id = f'{dirname(__file__)}\\smsmodel'
#     # config = PeftConfig.from_pretrained(peft_model_id)
#     # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
#     # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     # # Load the Lora model
#     # model = PeftModel.from_pretrained(model, peft_model_id)
#     batch = sms_tokenizer(prompt, return_tensors='pt')
#     # batch = batch.to(torch.device('cuda'))
#     # with torch.cuda.amp.autocast():
#     output = sms_model.generate(**batch, max_new_tokens=360)

#     res = sms_tokenizer.decode(output[0])
#     res = "<br />".join(res.split("\n"))
#     res = res.replace('<s>', '')
#     res = res.replace('</s>', '')
#     res = res.split('.', 1)[-1]

#     if '[/INST]'  in res:
#         res = res.split('[/INST]')[0]
#     elif '[/]' in res:
#         res = res.split('[/]')[0]
#     else:
#         res = res

#     prepared_response = { 
#         "type" : response_type, 
#         "data" : res, 
#     } 
#     # del model
#     # del tokenizer
#     # gc.collect()
#     # torch.cuda.empty_cache() 
#     return prepared_response
  

# def generate_email(hedis_measure, from_age, to_age, response_type, prompt):
#     # load model
#     # model_name = f'{dirname(__file__)}\\smsmodel'
#     # peft_model_id = f'{dirname(__file__)}\\smsmodel'
#     # config = PeftConfig.from_pretrained(peft_model_id)
#     # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
#     # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     # # Load the Lora model
#     # model = PeftModel.from_pretrained(model, peft_model_id)
#     batch = sms_tokenizer(prompt, return_tensors='pt')
#     # batch = batch.to(torch.device('cuda'))
#     # with torch.cuda.amp.autocast():
#     output = sms_model.generate(**batch, max_new_tokens=360)
#     res = sms_tokenizer.decode(output[0])
#     res = "<br />".join(res.split("\n"))
#     res = res.replace('<s>', '')
#     res = res.replace('</s>', '')
#     res = res.split('.', 1)[-1]

#     prepared_response = { 
#         "type" : response_type, 
#         "data" : res, 
#     } 
#     # del model
#     # del tokenizer
#     gc.collect()
#     # torch.cuda.empty_cache() 
#     return prepared_response

def generate_ivr(hedis_measure, response_type, prompt):
    
    prompt = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=ivr_model, prompt=prompt)

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
            prompt = f'Please write a sms for informing customer about {hedis_measure} whose  age From {from_age} To {to_age}'
            # data = generate_sms(hedis_measure, from_age, to_age, response_type, prompt)
        elif response_type == 'email':
            prompt = f'Please write a email for informing customer about {hedis_measure} whose  age From {from_age} To {to_age}'
            # data = generate_email(hedis_measure, from_age, to_age, response_type, prompt)
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