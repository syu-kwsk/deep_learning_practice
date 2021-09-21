import os
import urllib.request
import zipfile
import tarfile
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

file_id = '1JZi_XRrvAGlZ6ABCKB7oBtp8JCrSBfqr'
destination = './weights/pspnet50_ADE20K.pth'

if not os.path.exists(destination):
    print('学習済みモデル「pspnet50_ADE20K.pth」をダウンロードします。')
    download_file_from_google_drive(file_id, destination)

file_id = '1PoHxv2ZKuQlsyNbEXQ0CPa7r2xXtoFrL'
destination = './weights/pspnet50_30.pth'

if not os.path.exists(destination):
    print('学習済みモデル「pspnet50_30.pth」をダウンロードします。')
    download_file_from_google_drive(file_id, destination)

data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

file_id = '1wuKneBJGMGClQt29gUyAH1Pl1HsNwUgA'
destination = './data/image.jpg'
download_file_from_google_drive(file_id, destination)

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

if not os.path.exists(target_path):
    print('VOC2012のデータセットをダウンロードします。')
    urllib.request.urlretrieve(url, target_path)

    tar = tarfile.TarFile(target_path)
    tar.extractall(data_dir)
    tar.close()

print('ダウンロードが終了しました。')
