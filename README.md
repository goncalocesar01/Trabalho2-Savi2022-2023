# Trabalho2-Savi2022-2023
Aqui está apresentado em repositório a resposta ao Trabalho 2 de Savi 2022-2023

https://github.com/miguelriemoliveira/savi_22-23/tree/main/Trabalho2
 
Objetivos

Objetivo 1 - Treino de um classificador em deep learning

Objetivo 2 - Pre-processamento 3D

Objetivo 3 - Classificação de objetos na cena

Objetivo 4 - Descrição áudio da cena

Objetivo 5 - Sistema em tempo real

Como resposta a estes objetivos realizamos este programa que é corrido apartir do main, este vai selecionar os outros scripts apartir do input dado ao terminal
No input 1 corre a parte do treino do classificado 

# Objetivo 2
O sistema processa a nuvem de pontos de uma cena (dados do RGB-D Scenes Dataset), isola os objetos na nuvem de pontos, encontra uma transformação geométrica em que coloqua o sistema de coordenadas no centro da mesa com o eixo Z virado para baixo. A nova nuvem de pontos contem apenas os pontos contidos numa caixa próxima da origem, e dessa forma isola os pontos da mesa.
Usando o plano da mesa usando RANSAC, separa os pontos da mesa dos pontos dos objeto. Foi desenvolvida uma aplicação em open3d que permita visualizar os nomes dos objetos e o resultado da deteção.
O sistema devolve várias propriedades dos objetos como:

a altura,

a largura,

o volume,

e a distancia ao centro.
