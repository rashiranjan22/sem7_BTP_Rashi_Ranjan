
from trankit import Pipeline


# initialize a multilingual pipeline
p = Pipeline(lang='hindi', gpu=True, cache_dir='./cache')



# Tokenizing an English input
en_output = p.tokenize(''' इसके अतिरिक्त गुग्गुल कुंड, भीम गुफा तथा भीमशिला भी दर्शनीय स्थल हैं ।''')
print(en_output)
en_output = p.posdep(''' इसके अतिरिक्त गुग्गुल कुंड, भीम गुफा तथा भीमशिला भी दर्शनीय स्थल हैं ।''')
print(en_output)
en_output = p.lemmatize(''' इसके अतिरिक्त गुग्गुल कुंड, भीम गुफा तथा भीमशिला भी दर्शनीय स्थल हैं ।''')
print(en_output)



