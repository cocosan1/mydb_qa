import streamlit as st
import os
import requests
# from notion_client import Client
import os
import shutil

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI


from llama_index import download_loader

# pip install streamlit llama-index langchain langchain_community

st.markdown('### 社内Q&A ')

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

integration_token = st.secrets['NOTION_INTEGRATION_TOKEN']

# embeddingモデルの初期化
embedding_model = OpenAIEmbeddings() 

# llmの初期化
llm = ChatOpenAI(temperature=0) # temperature responseのランダム性0

##########################　notionからテキストデータの取得 関数内で使用
def make_doc(notion_token):
    
    ############　page idの自動取得
    url = f"https://api.notion.com/v1/search"

    headers = {
        "accept": "application/json",
        "Notion-Version": "2022-06-28",
        "Authorization": f"Bearer {notion_token}"
    }

    json_data = {
        # タイトルを検索できる
        #"query": "ブログ",
        # 絞り込み(データベースだけに絞るなど)
        #"filter": {
        #    "value": "database",
        #    "property": "object"
        #},
        # ソート順
        "sort": {
            "direction": "ascending",
            "timestamp": "last_edited_time"
        }
    }

    response = requests.post(url, json=json_data, headers=headers)
    j_response = response.json()
    j_response1 = j_response['object']
    j_response2 = j_response['results']

    #page_idの取得
    page_ids = []
    for page in j_response2:
        page_id = page["id"]
        page_ids.append(page_id)
    
    NotionPageReader = download_loader('NotionPageReader')

    documents = NotionPageReader(integration_token=notion_token).load_data(page_ids=page_ids)

    return documents

raw_documents = make_doc(integration_token)

st.write(raw_documents)

### CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",  # セパレータ
    chunk_size = 300,  # チャンクの文字数
    chunk_overlap = 0,  # 重なりの最大文字数
    length_function = len, # チャンクの長さがどのように計算されるか len 文字数 
    is_separator_regex = False, 
    # セパレータが正規表現かどうかを指定 True セパレータは正規表現 False 文字列
)

## データの分割
# documentオブジェクト
documents = text_splitter.split_documents(raw_documents)

# テキストデータ　M25Retrieverに渡す前にオブジェクトから文字列に変換する必要あり。
documents_txt = [doc.text for doc in text_splitter.split_documents(raw_documents)]


#1次情報cbのindex化
def save_fiass():
    ##############notionからテキストデータの取得

    dir_path = './fiass_index/'

    if os.path.isdir(dir_path):
        # fiass_indexフォルダの削除
        shutil.rmtree(dir_path)
        st.write(f'{dir_path} 削除完了')

        # vectorstoreの作成
        vectorstore = FAISS.from_documents(documents, embedding_model)
        # vectorstoreの保存
        vectorstore.save_local("./fiass_index")
        st.write('vectorstoreの保存完了')

    else:
        # vectorstoreの作成
        vectorstore = FAISS.from_documents(documents, embedding_model)
        # vectorstoreの保存
        vectorstore.save_local("./fiass_index")
        st.write('vectorstoreの保存完了')



def main():
    # function名と対応する関数のマッピング
    funcs = {
        # 'retrieverの実行': run_retriever,
        'fiass_vectorstoreの作成': save_fiass
    }

    selected_func_name = st.sidebar.selectbox(label='項目の選択',
                                             options=list(funcs.keys()),
                                             key='func_name'
                                             )
            
    # 選択された関数を呼び出す
    render_func = funcs[selected_func_name]
    render_func()

if __name__ == '__main__':
    main()
