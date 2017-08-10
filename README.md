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
        Tabelas em formato csv
    logs: Arquivos .log com os logs de execução dos módulos
    xml: Arquivos .xml (e .dtd) com os dados de entrada do exercício
    
Nomenclatura das avaliações
    
    (medida)_(tipo_documento), onde medida é pxr para os dados de Precisão x Recall, r para
    Precision-R(A,B) e eval para o resto. Tipo_documento define se é um gráfico (graph) 
    ou uma tabela (table)
    
Bibliotecas externas usadas:

    LXML (http://lxml.de/index.html), especificamente a api etree para parsing de arquivos xml verificados por dtd
    NLTK (http://www.nltk.org/), especificamente a função word_tokenize para transformar um texto numa lista de palavras
    MatPlotLib (https://matplotlib.org/), especificamente a plotagem de gráficos
    
Comentários sobre os arquivos de configuração:

    De modo a permitir gerar uma versão de resultados esperados e obtidos usando PorterStemmer
    e outra sem usá-lo, os arquivos busca.cfg, gli.cfg e index.cfg foram alterados, adicionando 
    as configurações RESULTADOS_STEM, ESCREVA_STEM e LEIA_STEM.
    
Comentários sobre o BPREF:
    
    Segundo o artigo original (dl.acm.org/citation.cfm?id=1009000), a formula usada é 
    1/R *  somatório(1 - (numero de resultados irrelevantes até o momento/ R), sendo apenas 
    somados os resultados relevantes, e sendo R  o total de documentos relevantes à query no
    corpus. No entanto, segundo (http://icb.med.cornell.edu/wiki/index.php/BPrefTrecEval2006)
    e confirmado no estudo para esse trabalho, é necessário transformar o numerador em
    min(numero de resultados irrelevantes até o momento, R) de modo a garantir que quando os
    resultados irrelevantes ultrapassam R, o sistema para de considerá-los, e não retorna 
    probabilidades de -10 ou coisas assim. O denominador também deveria ser mudado para min(N, R), 
    sendo N o numero de documentos não relevantes, mas dado a quantidade de artigos no corpus
    e a pouca quantidade de documentos relevantes é seguro afirmar que R << N
    