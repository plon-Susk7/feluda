import os, sys, json, datetime, copy, uuid, requests
from dotenv import load_dotenv
load_dotenv()
import logging
from flask import Flask, request, jsonify 
from flask_cors import CORS
from pymongo import MongoClient
from io import BytesIO
import skimage, PIL
import numpy as np
from monitor import timeit
from elasticsearch import Elasticsearch
from elasticsearch import helpers as eshelpers
from VideoAnalyzer import VideoAnalyzer, compress_video
from analyzer import ResNet18, detect_text, image_from_url, doc2vec, detect_lang
from search import ImageSearch, TextSearch, DocSearch
from helper import index_data, get_text_vec, get_image_vec, get_vid_vec
# from send import add_job_to_queue
import cv2
from indices import check_index
from datetime import datetime
import wget
from services.es import get_es_instance

imagesearch = ImageSearch()
docsearch = DocSearch()
textsearch = TextSearch()
resnet18 = ResNet18()

application = Flask(__name__)
CORS(application)

logger = logging.getLogger("tattle-api")

mongo_url = os.environ['MONGO_URL']
cli = MongoClient(mongo_url)
db = cli[os.environ.get("DB_NAME")]
coll = db[os.environ.get("DB_COLLECTION")]

es = get_es_instance()

es_vid_index = os.environ['ES_VID_INDEX']
es_img_index = os.environ['ES_IMG_INDEX']
es_txt_index = os.environ['ES_TXT_INDEX']

# Create ES indices if they don't exist
check_index(es, es_vid_index, index_type="video")
check_index(es, es_img_index, index_type="image")
check_index(es, es_txt_index, index_type="text")

@application.route('/health')
def health_check():
    logger.debug('<health-check>')
    return "OK"

@application.route('/media', methods=['POST'])
def media():
    try:
        data = request.get_json(force=True)
        print(data)
        print("Adding requested job to indexing queue ...")
        add_job_to_queue(data)
        print("Job added")
        return 'media enqueued', 200
    except Exception as e:
        return 'Error indexing media : '+str(e), 500


@timeit
@application.route('/search', methods=['POST'])
def search():
    data = request.get_json(force=True)

    def query_es(index, vec):
        if type(vec) == np.ndarray:
            vec = vec.tolist()
        if index == es_txt_index:
            calculation = "1 / (1 + l2norm(params.query_vector, 'text_vec'))"
        elif index == es_img_index:
            calculation = "1 / (1 + l2norm(params.query_vector, 'image_vec'))"
        elif index == es_vid_index:
            calculation = "1 / (1 + l2norm(params.query_vector, 'vid_vec'))"

        q = {
        "size": 10,
            "query": {
                "script_score": {
                    "query" : {
                        "match_all" : {}
                        },
                    "script": {
                        "source": calculation, 
                        "params": {"query_vector": vec}
                        }
                    }
                }
            }

        resp = es.search(index=index, body = q)
        res = parse_response(resp)
        return res
        
    def find_similar_text(index, text):
        resp = es.search(
            index=index, 
            body = {"query": {
                        "match": {
                            "text": text}}})
        res = parse_response(resp)
        return res

    def parse_response(resp):
        doc_ids, dists, source_ids, sources, texts = [], [], [], [], [] 

        for h in resp['hits']['hits']:
            doc_ids.append(h['_id'])
            dists.append(h['_score'])
            source_ids.append(h["_source"]["source_id"])
            sources.append(h["_source"]["source"])
            texts.append(h["_source"].get("text", None))

        result = [{'doc_id' : doc_ids[i], 'dist' : dists[i], 'source' : sources[i], 'source_id': source_ids[i], 'text': texts[i]} for i in range(len(doc_ids))]
        return result

    if data["media_type"] == "text": # add error handling
        text = data["text"]
        query_vec = get_text_vec(text)
        vec_search_result = query_es(index=es_txt_index, vec=query_vec)
        text_search_result = find_similar_text(index=es_txt_index, text=text)
        return jsonify(vec_search_result, text_search_result)

    elif data["media_type"] == "image":
        image_url = data['file_url']
        query_vec = get_image_vec(image_url)
        result = query_es(index=es_img_index, vec=query_vec)
        return jsonify(result)

    elif data["media_type"] == "video":
        video_url = data["file_url"]
        vid_analyzer = get_vid_vec(video_url)
        query_vec = vid_analyzer.get_mean_feature().tolist()
        result = query_es(index=es_vid_index, vec=query_vec)
        return jsonify(result)

@application.route('/find_text', methods=['POST'])
def find_text():
    data = request.get_json(force=True)
    image_url = data.get('image_url')
    image_dict = image_from_url(image_url)
    return jsonify(detect_text(image_dict['image_bytes']))

@application.route('/delete_doc', methods=['POST'])
def delete_doc():
    data = request.get_json(force=True)
    doc_id = data.get('source_id')
    ret = {}
    if not doc_id:
        ret['failed'] = 1
        ret['error'] = 'no doc id provided'
        return jsonify(ret)

    result = coll.delete_one({"doc_id" : doc_id})
    if result.deleted_count == 1:
        ret['failed'] = 0
    else:
        ret['failed'] = 1
        ret['error'] = 'no matching document'
    return jsonify(ret)

@application.route('/search_tags', methods=['POST'])
def search_tags():
    ret = {}
    data = request.get_json(force=True)
    tags = data.get('tags')
    if not tags:
        ret['failed'] = 1
        ret['error'] = 'no tags provided'
        return jsonify(ret)

    sources = data.get('sources')
    if sources:
        query = {"tags.value" : {"$in" : tags}, "tags.source" : {"$in" : sources}}
    else:
        query = {"tags.value" : {"$in" : tags}}

    docs = []
    for doc in coll.find(query):
        docs.append(doc.get('doc_id'))

    ret['docs'] = docs
    ret['failed'] = 0
    return jsonify(ret)

@application.route('/update_tags', methods=['POST'])
def update_tags():
    """
    tags are stored as follows:
    'tags' : [{'value' : 'politics' , 'source' : 'tattle'},
              {'value' : 'election' , 'source' : 'TOI'   },
              {'value' : 'politics' , 'source' : 'altnews'}]

    update_tags will only have a list of tags ['election','politics'] and a source : 'tattle'
    it'll create two tags: [{'value' : 'election' , 'source' : 'tattle'}, {'value' : 'politics', 'source' : 'tattle'}]
    """
    data = request.get_json(force=True)
    doc_id = data.get('doc_id')
    tags = data.get('tags') # e.g. [{'tag' : 'politics', 'source' : 'tattle'}, {'tag' : 'election', 'source' : 'TOI'}
    source = data.get('source') or 'tattle'
    if doc_id is None:
        ret = {'failed' : 1, 'error' : 'no doc_id provided'}
    elif tags is None:
        ret = {'failed' : 1, 'error' : 'no tags provided'}
    else:
        doc = coll.find_one({"doc_id" : doc_id})
        if doc is None:
            ret = {'failed' : 1, 'error' : 'doc not found'}
        else:
            new_tags = [{'value' : t, 'source':source} for t in tags]
            all_tags = new_tags + doc.get('tags', [])
            updated_tags = list({(t['value'],t['source']):t for t in all_tags}.values())

            date = datetime.datetime.now()
            coll.update_one({"doc_id" : doc_id}, {"$set" : 
                {"tags" : updated_tags, "date_updated" : date}})
            ret = {'failed' : 0}
    return jsonify(ret)

@application.route('/remove_tags', methods=['POST'])
def remove_tags():
    data = request.get_json(force=True)
    doc_id = data.get('doc_id')
    tags = data.get('tags')
    source = data.get('source')
    if doc_id is None:
        ret = {'failed' : 1, 'error' : 'no doc_id provided'}
    elif tags is None:
        ret = {'failed' : 1, 'error' : 'no tags provided'}
    elif source is None:
        ret = {'failed' : 1, 'error' : 'no source provided'}
    else:
        doc = coll.find_one({"doc_id" : doc_id})
        if doc is None:
            ret = {'failed' : 1, 'error' : 'doc not found'}
        else:
            to_remove_tags = [{'value' : t, 'source':source} for t in tags]
            updated_tags = doc.get('tags',[])
            removed = []
            for tag in to_remove_tags:
                if tag in updated_tags:
                    updated_tags.remove(tag)
                    removed.append(tag)
            date = datetime.datetime.now()
            coll.update_one({"doc_id" : doc_id}, {"$set" : 
                {"tags" : updated_tags, "date_updated" : date}})
            ret = {'failed' : 0, 'removed_tags':removed}
    return jsonify(ret)

def analyze_image(image_url):
    image = skimage.io.imread(image_url)
    image = PIL.Image.fromarray(image)
    embedding = get_image_embedding(image)

@application.route('/upload_text', methods=['POST'])
def upload_text():
    print(request.get_data())
    data = request.get_json(force=True)
    res = index_data(es, data)
    return jsonify(res)

@application.route('/upload_video', methods=['POST'])
def upload_video():
    print(request.get_data())
    data = request.get_json(force=True)
    res = index_data(es, data)
    return jsonify(res)

@application.route('/upload_image', methods=['POST'])
def upload_image():
    print(request.get_data())
    data = request.get_json(force=True)
    res = index_data(es, data)
    return(jsonify(res))
 
if __name__ == "__main__":
    application.run(host="0.0.0.0", port=7000, debug=True)
