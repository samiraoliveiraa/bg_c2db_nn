# Band Gap Prediction - Trabalho Final de Redes Neurais
_Grupo: Natália Alcantara, Samira Oliveira e Geovana Bettero_

Este repositório trata-se de um trabalho final feito por três estudantes da ILUM Escola de Ciência. O trabalho se concentra na criação de uma rede neural MLP para prever o band gap de um material por meio de um banco de dados chamado (C2DB) de materiais 2D. O banco de dados contém propriedades estruturais, termodinâmicas, elásticas, eletrônicas, magnéticas e ópticas de cerca de 4.000 materiais bidimensionais distribuídos em mais de 40 estruturas cristalinas diferentes. Para montagem dataset foram escolhidas algumas colunas do banco para sua formulação, como visto a seguir.

<hr>
<b><br>Importância<br></b>
Prever o band gap de um material é de grande importância em várias áreas da ciência, principalmente na engenharia de materiais e na indústria eletrônica. O band gap é uma propriedade fundamental que determina o comportamento de um material. Conhecer o band gap de um material é crucial para projetar novos materiais com propriedades específicas, como semicondutores para dispositivos eletrônicos. Desta forma, com redes neurais tornou-se possível prever o band gap de materiais com maior precisão e eficiência, facilitando o desenvolvimento de materiais com propriedades sob medida para diversas aplicações.

<img src="https://nirajchawake.wordpress.com/wp-content/uploads/2014/10/picture1.png" width="400">

<hr>
<b><br>Informações sobre o Dataset<br></b>
O Dataset em questão possui os seguintes atributos:


_Fórmula:_<br>A fórmula química  influencia o band gap através de sua composição atômica. Esse atributo também foi usado para prever a eletronegatividade do material por meio de técnicas de partição para extração dos átomos individuais. 

_Thermodynamic stability level:_<br>  A estabilidade termodinâmica de um material afeta diretamente suas propriedades físicas e eletrônicas. Materiais mais estáveis tendem a ter uma estrutura cristalina mais ordenada, o que pode influenciar o band gap.

_Energy(ev/atom):_<br>  A energia total do sistema  pode fornecer informações sobre a força das ligações químicas presentes no material, o que influencia as propriedades eletrônicas.

_Work function(eV):_<br> A função de trabalho é a energia mínima necessária para remover um elétron de um material para o vácuo. Ela está  relacionada à energia de Fermi do material e pode afetar o comportamento dos elétrons na superfície do material.

_Heat of formation(eV/atom):_<br> O calor de formação é a quantidade de energia liberada ou absorvida quando um composto é formado a partir de seus elementos constituintes. Ele pode indicar a estabilidade do material e sua capacidade de formar ligações químicas.

_Space group number:_<br>  O grupo espacial descreve a simetria da estrutura cristalina do material, podendo afetar a dispersão de elétrons e lacunas na estrutura de bandas. Seu número vai de 1 a 230 e cada grupo representa uma simetria diferente de um cristal periódico.

_Volume of unity cell(Å³):_<br> O volume da célula unitária está diretamente relacionado à densidade do material e à distância média entre os átomos.

_Eletronegativity:_<br> A eletronegatividade é a tendência de um átomo de atrair elétrons para si mesmo quando está ligado a outro átomo, o que pode influenciar a polaridade das ligações químicas.

_Band gap:_<br> O band gap é a energia necessária para excitar um elétron de um estado ligado para um estado não ligado (condução) e é o alvo de previsão.

_Elementos presentes nos materiais do dataset:_<br> Esses elementos parcionados podem ser utilizados para previsões de outros materiais que não estão no dataset.

<hr>
<b><br> Funcionamento Código e Utilização <br></b>

O código deste repositório emprega uma rede neural MLP utilizando o PyTorch Lightning. Nos notebooks fornecidos, há uma explicação detalhada do processo, abrangendo desde o tratamento dos dados até o treinamento da rede e a otimização utilizando Optuna. As etapas do processo são descritas no notebook, fornecendo detalhes sobre preparação dos daods,parcionamento das fórmulas para criar o atributo de eletronegatividade e como a rede neural é ajustada para melhor desempenho. Este repositório serve como um recurso valioso para aqueles que desejam entender e aplicar métodos de aprendizado de máquina para previsão de band gap em materiais 2D.

<b><br>Bibliotecas<br></b>
As Bibliotecas e funções usadas e necessárias para importação: Pandas, re, lightnig, matplotlib, numpy, pickle, torch, scipy, sklearn e optuna.  

<hr>

<b>Documentos no Github<br></b>

_C2DB_full.csv_: Dataset completo

_dataset_tratado.csv_: Dataset tratado

_tratamento_de_dados.ipynb_: Processo de tratamento do dataset

_treinamento_da_rede_otimizacao_gpu.ip_: Implementação da rede neural

_otimizacao_dos_hiperparametros.py_: Processo de otimizacao dos hiperparâmetros utilizando Optuna

_trials_job_1516.out_:Este script é utilizado para executar tarefas de treinamento e otimização em um ambiente de computação que suporte GPUs 

_modelo_final.BG.p_: arquivo que contém os pesos e vieses da rede neural treinada


<hr>
<b><br>Conclusão<br></b>
Após a otimização, nota-se que o conjunto ideal hiperparâmetros que se adapta a rede resultou em um Erro Quadrático Médio (RMSE) de aproximadamente 0,867. Dessa forma é possível concluir a rede em questão não é tão satisfatória, por o target necessitar de uma alta precisão nos resultados, indicando assim a necessidade de aprimoramentos adicionais na rede. 

<hr>
<b><br>Referências<br></b>

[1] Banco de dados C2DB: https://cmr.fysik.dtu.dk/c2db/c2db.html#brief-description 

[2] CASSAR, D. R. PyTorch Lightning. (2024)

[3] CASSAR, D. R.Treinando uma rede neural com pytorch. (2024)

[4] CASSAR, D. R. Redes neurais artificiais do zero em Python. (2024)

[5] CASSAR, D. R. Avaliação de modelos: a estratégia de divisão entre treino e teste. (2023)

[6] CASSAR, D. R. Transformação e normalização. (2023)

[7] CASSAR, D. R. Conversão simbólico-numérico. (2023)

[8] ChatGPT para ajuda na resolução de bugs

