import streamlit as st
import os
import requests
# from notion_client import Client
import os
import shutil
import logging
import pickle

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain.document_transformers import LongContextReorder
from langchain.chains import LLMChain

from llama_index import download_loader

# pip install streamlit llama-index langchain langchain_community rank_bm25

st.markdown('### myDB Q&A ')

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
    # with st.expander('j_response',expanded=False):
    #     st.write(j_response)

    j_response1 = j_response['object']
    j_response2 = j_response['results']

    #page_nameの取得
    urls = []
    for idx, page in enumerate(j_response2):
        url =j_response2[idx]['url']
        urls.append(url)

    #page_idの取得
    page_ids = []
    for page in j_response2: # pegeオブジェクトを回す
        page_id = page["id"]
        page_ids.append(page_id)

    NotionPageReader = download_loader('NotionPageReader', custom_path="local_dir")
    # custom_path="local_dir" これがないとstreamlit cloudで動かなくなる

    documents = NotionPageReader(integration_token=notion_token).load_data(page_ids=page_ids)

    return documents, urls

##########################　notionから取得したdocumentsの分割
def split_doc():
    raw_documents, urls = make_doc(integration_token)

    ### llamaindex型のオブジェクトからlangchain型のオブジェクトに変換
    documents = []
    for doc, url in zip(raw_documents, urls):
        # 要素の抽出
        page_content = doc.text
        metadata = doc.metadata
        # langchain型のオブジェクトに変換
        document = Document(page_content=page_content, metadata=metadata)

        # 追加するmetadataを定義
        new_metadata = {"url": url}

        # documentのmetadataに追加する
        document.metadata.update(new_metadata)
        # listに追加
        documents.append(document)

    ### CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator = "\n",  # セパレータ
        chunk_size = 200,  # チャンクの文字数
        chunk_overlap = 0,  # 重なりの最大文字数
        length_function = len, # チャンクの長さがどのように計算されるか len 文字数 
        is_separator_regex = False, 
        # セパレータが正規表現かどうかを指定 True セパレータは正規表現 False 文字列
    )

    ## データの分割
    # documentオブジェクト
    splitted_documents = text_splitter.split_documents(documents)

    # with st.expander('splitted_documents', expanded=False):
    #             st.write(splitted_documents)

    # テキストデータ　M25Retrieverに渡す前にオブジェクトから文字列に変換する必要あり。
    splitted_documents_txt = [doc.page_content for doc in text_splitter.split_documents(documents)]

    return splitted_documents, splitted_documents_txt


def run_retriever():
    # vectorstoreの読み込み
    vectorstore = FAISS.load_local("./fiass_index", embedding_model)

    # splitted_documents_txtの読み込み
    with open("splitted_documents_txt.pkl", "rb") as f:
        splitted_documents_txt = pickle.load(f)

    ## 初期化 retriever
    # bm25 retriever
    bm25_retriever = BM25Retriever.from_texts(splitted_documents_txt)
    bm25_retriever.k = 2

    # fiass_retriever
    fiass_retriever = vectorstore.as_retriever(
         embedding_function=OpenAIEmbeddings(), 
         search_kwargs={"k": 2}
         )
    
    # MultiQueryRetrieverの初期化
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )

    with st.expander('設定: retrieverの比率', expanded=False):
        ## retrieverの比率設定
        # bm25_retriever
        bm25_rate = st.slider('■ bm25_retrieverの比率設定', 
                                min_value=0.0, 
                                max_value=1.0,
                                value=0.2, 
                                step=0.1
                                ) 
        # fiass_retriever
        fiass_rate = st.slider('■ fiass_retrieverの比率設定', 
                                min_value=0.0, 
                                max_value=1.0,
                                value=0.5, 
                                step=0.1
                                )

        val_c = st.slider('■ 上位ランク下位ランクバランス調整', 
                            min_value=0, 
                            max_value=100,
                            value=80, 
                            step=10
                            ) 

        # multiquery_retriever
        multiquery_rate = 1- bm25_rate - fiass_rate

        st.write(f'-bm25: {bm25_rate}- -fiass: {fiass_rate}- multi_query: {multiquery_rate}')

    # ensemble retrieverの設定
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, fiass_retriever, multiquery_retriever], 
        weights=[bm25_rate, fiass_rate, multiquery_rate],
        c=val_c,
        _='''
        複数の検索結果を統合する際に、上位にランク付けされた項目と下位にランク付けされた項目の
        バランスを調整するための定数。
        「c」の値が大きいほど、上位にランク付けされた項目がより強く優先される。
        '''
    )

    # promptのtemplateの作成
    template = """
    ###Instruction###
    ・あなたはが優秀なAIです。
    ・私たちはあなたに以下の「コンテキスト情報」を与えます。
    ・あなたのタスクは「質問」に対して「コンテキスト情報」を基に的確に答えることです。
    ・質問に対して的確な良い返答には30万ドルのチップを渡します！
    ・必ず日本語で答えなければなりません。
    
    ###Question### 
    {question}

    ###Context###
    {context}

    """
    # promptオブジェクトの作成　チャットモデルに情報と質問を構造化された方法で提示
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)

    
    # chat_input
    query = st.chat_input("ご用件をどうぞ 「・・・説明して」")

    if query:

        # ensemble_retrieverの実行
        ensemble_docs = ensemble_retriever.get_relevant_documents(query)
        # クエリをベクトル化する際に、OpenAI Embeddingsを使用

        with st.expander('設定: chunk数の設定', expanded=False):
            # chunk数を表示
            st.write(f'■ ensemble_docsのchunk数: {len(ensemble_docs)} ■ 全chunk数: {len(splitted_documents_txt)}')

            # ensemble_docsのchunkの数を決定
            len_chunk2 = st.slider('■ ensemble_docsのchunkの数の絞込み', 
                                min_value=0, 
                                max_value=10,
                                value=4, 
                                step=1
                                ) 

            # chunkの数を絞込み
            ensemble_docs = ensemble_docs[:len_chunk2]

            # chunk数を表示
            st.write(f'■ chunk数: {len(ensemble_docs)}')
            
        # multiquery_retrieverが生成したqueryをコマンドプロンプトにログ表示
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        
        # 精度を上げる為に関連性の低いドキュメントはリストの中央に。
        # 関連性の高いドキュメントは先頭または末尾に配置します。
        reordering = LongContextReorder()
        # ドキュメントを再配置
        reordered_docs = reordering.transform_documents(ensemble_docs)

        # LLMとPromptTemplateを連携させクエリを実行
        chain = LLMChain(llm=llm, prompt=prompt)

        # chainの実行
        response = chain({'question': query, 'context': ensemble_docs})
        
        # chat表示
        with st.chat_message("user"):
            st.write(query)

        message = st.chat_message("assistant")
        message.write(response['text'])
        
        with st.expander('ensemble_docs', expanded=False):
             st.write(ensemble_docs)
        
        # with st.expander('reordered_docs', expanded=False):
        #     st.write(reordered_docs)
        
        with st.expander('response.context ■ sourceの確認', expanded=False):
            st.write(response['context'])


#1次情報cbのindex化
def save_fiass():
    # notionから取得したdocumentsの分割
    splitted_documents, splitted_documents_txt = split_doc()

    # ファイルの保存
    with open("splitted_documents_txt.pkl", "wb") as f:
        pickle.dump(splitted_documents_txt, f)

    ##############notionからテキストデータの取得

    dir_path = './fiass_index/'

    if os.path.isdir(dir_path):
        # fiass_indexフォルダの削除
        shutil.rmtree(dir_path)
        st.write(f'{dir_path} 削除完了')

        # vectorstoreの作成
        vectorstore = FAISS.from_documents(splitted_documents, embedding_model)
        # vectorstoreの保存
        vectorstore.save_local("./fiass_index")
        st.write('vectorstoreの保存完了')

    else:
        # vectorstoreの作成
        vectorstore = FAISS.from_documents(splitted_documents, embedding_model)
        # vectorstoreの保存
        vectorstore.save_local("./fiass_index")
        st.write('vectorstoreの保存完了')

def check_data():
    # notionから取得したdocument
    raw_documents, urls = make_doc(integration_token)

    # notionから取得したdocumentsの分割
    splitted_documents, splitted_documents_txt = split_doc()

    # dataの表示
    with st.expander('raw_documents', expanded=False):
        st.write(raw_documents)
    
    # with st.expander('documents: langchain型に変換', expanded=False):
    #     st.write(documents)
    
    with st.expander('splitted_documents', expanded=False):
        st.write(splitted_documents)


def main():
    # function名と対応する関数のマッピング
    funcs = {
        'retrieverの実行': run_retriever,
        'fiass_vectorstoreの作成': save_fiass,
        'データの確認': check_data
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
