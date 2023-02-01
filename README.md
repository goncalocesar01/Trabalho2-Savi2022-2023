# Trabalho2-Savi2022-2023
Aqui está apresentado em repositório a resposta ao Trabalho 2 de Savi 2022-2023

https://github.com/miguelriemoliveira/savi_22-23/tree/main/Trabalho2
 
Objetivos

Objetivo 1 - Treino de um classificador em deep learning

Objetivo 2 - Pre-processamento 3D

Objetivo 3 - Classificação de objetos na cena

Objetivo 4 - Descrição áudio da cena

Objetivo 5 - Sistema em tempo real

# Objetivo 1
Para treinar o classificador utilizou-se o dataset retirado do link http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/ (aqui apenas retiramos os objetos que se encontram nas cenas e alguns adicionais para o teste do sistema em tempo real, separando este em 80% ara treino e os restantes testar apartir da visualização) e a ferramenta torch como sistema de deep learning. O dataset foi separado em 80% treino e 20%.

# Objetivo 2
O sistema processa a nuvem de pontos de uma cena (dados do RGB-D Scenes Dataset), isola os objetos na nuvem de pontos, encontra uma transformação geométrica em que coloqua o sistema de coordenadas no centro da mesa com o eixo Z virado para baixo. A nova nuvem de pontos contem apenas os pontos contidos numa caixa próxima da origem, e dessa forma isola os pontos da mesa.
Usando o plano da mesa usando RANSAC, separa os pontos da mesa dos pontos dos objeto. Foi desenvolvida uma aplicação em open3d que permita visualizar os nomes dos objetos e o resultado da deteção.
O sistema devolve várias propriedades dos objetos como:

a altura,

a largura,

o volume,

e a distancia ao centro.

# Objetivo 3
O programa abre a point cloud apenas de cada objeto de maneira a retirar uma print ao fechar a janela e passando então  pelo classificador realizado no objetivo 1. 
# Objetivo 4
Para a descrição aúdio da cena foi utilizado o Google Text-to-speech utilizando a identificação do classificador.
# Objetivo 5
Para o objetivo 5 não tínhamos à disposição uma câmara adequada. 
Assim, decidimos identificar a mão de um utilizador na imagem obtida em tempo real e, 
colocando uma "bounding box" na ponta do dedo indicador, extraímos uma imagem clicando na tecla 'p'. 
Esta imagem vai ser processada pelo nosso identificador para obtermos uma possível classificação do objeto.
