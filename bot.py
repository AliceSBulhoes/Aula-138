import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo: avaliações de filmes e suas categorias (positivo ou negativo)
documentos = [
    ("Ótimo filme, adorei!", "positivo"),
    ("Que filme horrível, péssimo!", "negativo"),
    ("Gostei muito do filme.", "positivo"),
    ("Não recomendo, muito chato.", "negativo"),
    ("Amei", "positivo")
    # Adicione mais exemplos de avaliações e suas categorias aqui
    ("Adorei o filme", "positivo")
    ("Lixo","negativo")
    ("Sem história", "negativo")
    ("Enredo interessante", "positivo")
    ("Personagens com pouco desenvolvimento","negativo")
    ("Modelo 3D horroso","negativo")
    ("Ambientação maravilhosa", "positivo")
    ("Do começo ao fim horrivel","negativo")
    ("Cenas de ação muito bem animadas","positivo")
    ("Meu filme favorito","positivo")
    ("Pior filme que vi na minha vida","negativo")
    ("Animação pessima","negativo")
    ("Trilha sonora inesquecível", "positivo")
    ("Atores não sabem atuar", "negativo")
    ("Atuação impecável","positivo")
    ("Cenários bem feitos", "positivo")
    ("Parece que foi feito em uma tela verde","negativo")
    ("Efeitos visuais são espetaculares", "positivo")
    ("Edição feita por uma criança", "negativo")
    ("Filmagem de qualidade", "positiva")
    ("Efeitos visuais pessimos", "negativo")
    ("Roteiro escrito por uma criança", "negativo")
    ("Roteiro impecável", "positivo")
    ("História bem contruida", "positivo")
]

# Embaralhe os dados
random.shuffle(documentos)

# Separe os dados em textos e categorias
textos = [texto for texto, categoria in documentos]
categorias = [categoria for texto, categoria in documentos]

# Vetorização dos textos usando a contagem de palavras (CountVectorizer)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Treine um modelo de classificação (Naive Bayes, por exemplo)
modelo = MultinomialNB()
modelo.fit(X, categorias)

# Função para fazer previsões com base no texto inserido pelo usuário
def fazer_previsao(texto):
    texto_transformado = vectorizer.transform([texto])
    categoria = modelo.predict(texto_transformado)[0]
    return categoria

# Exemplo de classificação de texto com entrada do usuário
while True:
    entrada_usuario = input("Digite uma avaliação de filme (ou 'sair' para encerrar): ")
    if entrada_usuario.lower() == "sair":
        break
    previsao = fazer_previsao(entrada_usuario)
    print(f"Previsão: {previsao}")
