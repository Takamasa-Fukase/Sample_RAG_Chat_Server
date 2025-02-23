from langchain.text_splitter import RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitterだとちゃんとchunk_size制限を下回るチャンクを作れる模様
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 暫定で500で設定
    chunk_overlap=20, # 暫定で20で設定
    # https://github.com/hwchase17/langchain/issues/1663#issuecomment-1469161790
    # この情報によると、separatorsに最低限["\n\n", "\n", " ", ""]を含めておかないと無限ループでエラーが起きる模様。
    separators=["\n\n", "\n", " ", "", '、', '。']
)