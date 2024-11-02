import gradio as gr
import pickle

with open('merges.pkl','rb') as merges_file:
    merges = pickle.load(merges_file)

with open('vocab.pkl', 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text):
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break 
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  i = 0
  newids = []
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  
  return newids

def getTokens(inputText):
    tokens = list(inputText.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break 
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    strTokens = []
    for token in tokens:
        strTokens.append(decode([token]))
    return tokens, strTokens

color_list = ['rgb(139, 162, 253)', 'rgb(254, 225, 190)', 'rgb(252, 175, 166)', 'rgb(219, 254, 210)', 'rgb(235, 233, 229)', 'rgb(211, 247, 187)', 'rgb(200, 251, 254)', 'rgb(159, 249, 191)', 'rgb(212, 212, 212)', 'rgb(167 243 208)', 'rgb(251 207 232)', 'rgb(243, 165, 240)', 'rgb(239 68 68)', 'rgb(254 205 211)', 'rgb(139, 152, 254)', 'rgb(241 245 249)']
def getColoredText(inputText, tokens, strTokens):
    i = 0
    txtLen = len(inputText)
    numTokens = len(strTokens)
    outputString = """<html><head><style>h2 { font-size: 40px;} sub { vertical-align: sub; font-size: 4px;} span { font-size:16px;}</style></head><body><div><h2>Total characters in text: """
    outputString = outputString + str(txtLen) + """</h2><h2>Token count: """
    outputString = outputString + str(numTokens) + """</h2><p> """
    for strToken in strTokens:
        colorNum = i % len(color_list)
        chosencolor = color_list[colorNum]
        outputString = outputString + "<span style='background-color:" + chosencolor + ";'> " + strToken + "</span>"
        i = i + 1
    outputString = outputString + """</p></body>"""
    return outputString

def tokenizeText(inputText):
    tokens, strTokens = getTokens(inputText)
    return(getColoredText(inputText, tokens, strTokens))
    
title = "Tokenizer for Hindi language from Scratch"
description = "Created Hindi Tokenizer from Scratch"
examples = [
    "चिन्नयरसाल में भारत के आन्ध्रप्रदेश राज्य के अन्तर्गत के कडप जिले का एक गाँव है।", 
    "करासीबूंगा, काफलीगैर तहसील में भारत के उत्तराखण्ड राज्य के अन्तर्गत कुमाऊँ मण्डल के बागेश्वर जिले का एक गाँव है।",
    "इस वाक्य का अनुवाद गूगल100 ट्रांसलेट द्वारा किया जा रहा है।😅",
    "कैफ़ भोपाली ने कई हिंदी फिल्मों में गीत लिखे, किन्तु 1972 में बनी पाक़ीज़ा उनकी यादगार फिल्म रही।",
    "छाया गठबंधन की हार के बाद विशेष Yamato बी दा बॉल्स की तलाश में है स्ट्राइक मदपान, जो सितारों शूटिंग से उत्पन्न कहा जाता है। रहस्यमय Haja के साथ एक लड़ाई के बाद और उसके स्ट्राइक फटका, ड्राइव शूटिंग प्राप्त करने, Yamato तो Gunnos, एक रंगरूट बी DaPlayer से मुलाकात की। वह अपने ही हड़ताल को Yamato पुराने मित्रों और प्रतिद्वंद्वियों के साथ विजेता प्रतियोगिता में भाग लेने शॉट मिल गया। लेकिन कुछ नहीं किया वे पता है।.. .. एक भयानक बुराई करने के लिए जोखिम में एक बार फिर बी दा विश्व डाल बारे में है।",
    "In 2019 google introduced BERT- Bidirectional Encoder Representations from Transformers (paper), which is designed to pre-train a language model from a vast corpus of rew text. What distinguishes it from existing word-embedding models like Word2vec, ELMo etc. is that it is a truly bidirectional model, meaning it is trained on unlabeled text by jointly conditioning both left and right context simultaneously."
]

demo = gr.Interface(
    tokenizeText, 
    inputs = [
        gr.Textbox(), 
    ],
    outputs = [
        gr.HTML(),
        ],
    title = title,
    description = description,
    examples = examples,
    cache_examples=False
)
demo.launch()