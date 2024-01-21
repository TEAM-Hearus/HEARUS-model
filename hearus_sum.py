import torch
from transformers import BertTokenizer
import pandas as pd
from konlpy.tag import Okt
import networkx as nx
import nltk
nltk.download('punkt')

# 저장된 모델 불러오기
sentence_model = torch.load('/content/drive/MyDrive/Model/hearus.pth')

# 토크나이저 초기화 (모델과 동일한 설정 사용)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 명사 추출 함수
def extract_nouns(text, stopwords):
    okt = Okt()
    nouns = okt.nouns(text)
    filtered_nouns = [noun for noun in nouns if noun not in stopwords and len(noun) > 1]
    return filtered_nouns

# 텍스트 랭크 알고리즘 적용 함수
def apply_text_rank(nouns):
    graph = nx.Graph()
    graph.add_nodes_from(nouns)
    for i in range(len(nouns)):
        for j in range(i + 1, len(nouns)):
            graph.add_edge(nouns[i], nouns[j])
    scores = nx.pagerank(graph)
    return scores

# 입력 생성 함수
def create_input(text, tokenizer):
    sentences = nltk.sent_tokenize(text)
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=35, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks, sentences

# 중요한 문장 추출 함수
def extract_important_sentences(text, model, tokenizer, num_sentences=2):
    model.eval()
    with torch.no_grad():
        input_ids, attention_masks, sentences = create_input(text, tokenizer)
        logits = model(input_ids, attention_masks)
        scores = torch.sigmoid(logits).squeeze().tolist()
        important_sentence_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        important_sentences = [sentences[i] for i in important_sentence_indices]
        return important_sentences


# JSON 파일 경로
file_path = '/content/drive/MyDrive/fine/computer.json'

# JSON 파일을 데이터프레임으로 불러오기
df = pd.read_json(file_path)

# 'sumText' 컬럼의 첫 번째 값을 사용하여 새로운 데이터프레임 생성
first_sumText = df['sumText'].iloc[0]
single_row_df = pd.DataFrame([first_sumText], columns=['sumText'])

# 'sumText' 컬럼의 모든 항목을 결합하여 전체 텍스트 생성
full_text = ' '.join([' '.join(item) if isinstance(item, list) else item for item in df['sumText']])


# 결과를 저장합니다.
output_file_path = '/content/drive/MyDrive/fine/single_summary_output4.json'
single_row_df.to_json(output_file_path, force_ascii=False)

# JSON 파일을 불러옵니다.
file_path = '/content/drive/MyDrive/fine/single_summary_output4.json'
df = pd.read_json(file_path)

# 중요한 문장과 단어 추출
important_sentences = []
important_words_list = []

for text in df['sumText']:
    imp_sentences = extract_important_sentences(text, sentence_model, tokenizer, num_sentences=2)
    important_sentences.append(imp_sentences)

    nouns = extract_nouns(text, stopwords)
    scores = apply_text_rank(nouns)
    imp_words = sorted(scores, key=scores.get, reverse=True)[:5]
    important_words_list.append(imp_words)

# 결과 저장
df['important_sentence'] = important_sentences
df['important_words'] = important_words_list

# JSON 파일로 저장
output_file_path = '/content/drive/MyDrive/fine/summary_lstm.json'
df.to_json(output_file_path, force_ascii=False)



import pandas as pd
import numpy as np
from ast import literal_eval

class Example:
    def __init__(self, df_original, important_sentence, important_words):
        self.df_original = df_original
        self.important_sentence = self.tokenize_and_eval(important_sentence)
        self.important_words = self.tokenize_and_eval(important_words)

    def tokenize_and_eval(self, text):
        if isinstance(text, str):
            # 문자열을 공백을 기준으로 분할하고 리스트로 반환
            return text.split()
        else:
            # 이미 리스트 형태인 경우 그대로 반환
            return text

    def update_unProcessedText(self):
        l = []
        for text_list in self.df_original['unProcessedText']:
            if text_list[0] in self.important_sentence[0]:
                l.append('highlight')
            elif text_list[0] in set(self.important_words):
                l.append('comment')
            else:
                l.append('none')
        for ind, unprocessedtext in enumerate(self.df_original['unProcessedText']):
            unprocessedtext[1] = l[ind]

    def clear_other_rows(self):
        for col in self.df_original.columns:
            if col != 'unProcessedText':
                self.df_original.loc[1:, col] = np.nan

    def save_to_json(self, path):
        self.df_original.to_json(path, force_ascii=False)

# 중요 문장과 단어 추출
file_path = '/content/drive/MyDrive/fine/summary_lstm.json'
df = pd.read_json(file_path)
important_sentence = df.loc[0, 'important_sentence']
important_words = df.loc[0, 'important_words']

# 원본 DataFrame 로드
original_file_path = '/content/drive/MyDrive/fine/computer.json'
df_original = pd.read_json(original_file_path)

# Example 객체 생성 및 데이터 처리
exam1 = Example(df_original, important_sentence, important_words)
exam1.update_unProcessedText()
exam1.clear_other_rows()

# 수정된 DataFrame을 새로운 JSON 파일로 저장
output_file_path = '/content/drive/MyDrive/fine/updated100.json'
exam1.save_to_json(output_file_path)
