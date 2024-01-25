import streamlit as st
import os
import requests
# from notion_client import Client
import datetime


from llama_index import download_loader

# pip install streamlit llama-index

st.markdown('### 社内Q&A ')

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

integration_token = st.secrets['NOTION_INTEGRATION_TOKEN']

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

#1次情報cbのindex化
def save_fiass():
    ##############notionからテキストデータの取得
    documents = make_doc(integration_token)
    st.write(documents)


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
