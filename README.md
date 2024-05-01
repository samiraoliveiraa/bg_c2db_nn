# bg_c2db_nn - Trabalho Final Redes Neurais
_Grupo: Natália Alcantara, Samira Oliveira e Geovana Betero_

Este repositório trata-se de um trabalho final feito por três estudantes da ILUM Escola de Ciência. O trabalho se concentra na criação de uma rede neural para prever o band gap de um material por meio de banco de dados de material 2D (C2DB). O banco de dados contém propriedades estruturais, termodinâmicas, elásticas, eletrônicas, magnéticas e ópticas de cerca de 4.000 materiais bidimensionais (2D) distribuídos em mais de 40 estruturas cristalinas diferentes. Para montagem dataset foram elencadas algumas colunas do banco para sua formulação, como visto a seguir:

<b><br>_Informações sobre o Dataset_<br></b>
O Dataset em questão possui os seguintes atritubos:

_Formula:_<br>A fórmula química e a estrutura dos materiais bidimensionais influenciam o band gap através de sua composição atômica. Essas propriedades também foram usadas para prever a eletronegatividade do material por meio de técnicas de partição para extração dos átomos individuais. 

_Thermodynamic stability level:_<br>  A estabilidade termodinâmica de um material afeta diretamente suas propriedades físicas e eletrônicas. Materiais mais estáveis tendem a ter uma estrutura cristalina mais ordenada, o que pode influenciar o band gap.

_Energy(ev/atom):_<br>  A energia total do sistema está relacionada à sua estrutura e estabilidade. Ela pode fornecer informações sobre a força das ligações químicas presentes no material, o que influencia as propriedades eletrônicas.

_Work function(eV):_<br> A função de trabalho é a energia mínima necessária para remover um elétron de um material para o vácuo. Ela está intimamente relacionada à energia de Fermi do material e pode afetar o comportamento dos elétrons na superfície do material.

_Heat of formation(eV/atom):_<br> O calor de formação é a quantidade de energia liberada ou absorvida quando um composto é formado a partir de seus elementos constituintes. Ele pode indicar a estabilidade do material e sua capacidade de formar ligações químicas.

_Space group number:_<br>  O grupo espacial descreve a simetria da estrutura cristalina do material, podendo afetar a dispersão de elétrons e lacunas na estrutura de bandas. Seu número vai de 1 a 230 e cada grupo representa uma simetria diferente de um cristal periódico 

_Volume of unity cell(Å³):_<br> O volume da célula unitária está diretamente relacionado à densidade do material e à distância média entre os átomos. Isso influencia a interação entre os elétrons.

_Eletronegativity:_<br> A eletronegatividade é a tendência de um átomo de atrair elétrons para si mesmo quando está ligado a outro átomo. Ela pode influenciar a polaridade das ligações químicas.

_Band gap:_<br> O band gap é a energia necessária para excitar um elétron de um estado ligado para um estado não ligado (condução) e é o principal alvo de previsão.


<b><br>_Funcionamento Código_<br></b>
O codigo usa como modelo uma rede neural com (DETALHAR MAIS DO FUNCIONAMENTO, tratamento, otimização etc etc), sendo as Bibliotecas e funções usadas e necessárias para importação:

Pandas: Biblioteca fundamental para manipulação e análise de dados em Python. Utiliza estruturas de dados, como o DataFrame, facilitando operações como carregamento, limpeza e transformação de dados de forma eficiente.

<b><br>Referencias<br></b>
Banco de dados C2DB: https://cmr.fysik.dtu.dk/c2db/c2db.html#brief-description 

