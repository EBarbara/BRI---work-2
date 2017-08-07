# BRI---work-2

Trabalho 2 de Busca e Recuperação de Informações - Avaliação de engine de busca.

Armazenado em repositório github (https://github.com/EBarbara/BRI---work-2)

O PorterStemmer foi obtido diretamente do site do prof. Martin Porter (https://tartarus.org/martin/PorterStemmer/python.txt), como indicado na página do curso, tendo como única modificação a parte do código auto-executável ("if __name__ == '__main__':" em diante) ter sido comentada.

Instruções:

    Instalar as bibliotecas LXML, NLTK e MatPlotLib
    Executar o arquivo App.py
    Ver os resultados

Organização das pastas

    Código fonte e arquivos de descrição (readme e modelo) na raiz
    config: Arquivos .cfg para configurar os módulos
    csv: Arquivos .csv como resultados dos processamentos de busca
    evaluation: Arquivos derivados das avaliações
        Gráficos em formato pdf
        Tabelas e listas de medidas independentes em formato csv
        Arquivos com nome '[medida]_nostemmer_[tipo]' indicam que foram gerados em uma busca sem PorterStemmer
        Da mesma forma, arquivos com nome '[medida]_stemmer_[tipo]' indicam que foram gerados em uma busca utilizando PorterStemmer
    logs: Arquivos .log com os logs de execução dos módulos
    xml: Arquivos .xml (e .dtd) com os dados de entrada do exercício

Bibliotecas externas usadas:

    LXML (http://lxml.de/index.html), especificamente a api etree para parsing de arquivos xml verificados por dtd
    NLTK (http://www.nltk.org/), especificamente a função word_tokenize para transformar um texto numa lista de palavras
    MatPlotLib (https://matplotlib.org/), especificamente a plotagem de gráficos