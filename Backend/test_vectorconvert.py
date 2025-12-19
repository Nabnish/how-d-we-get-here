import vectorConvert, os

print('Index files exist:', os.path.exists(vectorConvert.INDEX_FILE), os.path.exists(vectorConvert.META_FILE))

try:
    vectorConvert.answer_with_pubmed('What is this about?')
    print('answer_with_pubmed returned')
except Exception as e:
    print('answer_with_pubmed raised:', type(e).__name__, e)

try:
    res = vectorConvert.build_index_from_text([{'text':'Hello world', 'title':'t', 'source':'s'}])
    print('build_index_from_text returned:', res)
except Exception as e:
    print('build_index_from_text raised:', type(e).__name__, e)
