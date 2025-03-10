# Clusterização de Padrões Atmosféricos associados a Extremos de Onda

Este repositório contém um conjunto de scripts Python desenvolvidos para identificar e analisar padrões atmosféricos associados a extremos de onda ao longo da costa sul e sudeste do Brasil. O trabalho baseia-se no uso do K-Means para clusterização dos padrões atmosféricos associados a extremos de onda. 

O Artigo publicado associado a esse algortimo encontra-se em:  https://link.springer.com/article/10.1007/s11069-025-07166-7

Os dados para teste estão dispoíveis em: https://drive.google.com/drive/folders/1hejo_AZWd4qkTQNJyEvG2YT9sy4-XVKm?hl=pt-br

## Estrutura do Repositório

Os scripts estão organizados no diretório `src` e cada um possui uma funcionalidade específica dentro do processo de clusterização:

- `boia_domain_disserta.py`: Seleciona e exporta as coordenadas do ponto selecionado e cria um mapa para região de estudo.
- `GetDays.py`: Seleciona os dias associados a cada WP ou a cada 24-48-72 horas antes do extremo. 
- `KneeLocWP.py`: Determina o número ótimo de clusters para o algoritmo K-Means usando o método do cotovelo.
- `mainWP.py`: Script principal que coordena o processo de clusterização, chamando as funções necessárias dos outros scripts.
- `PlotWP.py`: Gera visualizações dos padrões atmosféricos identificados, campo de onda extrema, histograma direcional no ponto selecionado, séries temporais dos eventos, distribuição dos WP. 
- `ProcessWP.py`: Processa os dados meteoceanograficos de forma adequada para clusterização temporal e para entrada no algoritmo do K-Means.

## Instalação

Dependências necessárias:

numpy
pandas
xarray
scikit-learn
matplotlib

## Uso
Para executar a análise completa, inicie o mainWP.py da seguinte forma:

python src/mainWP.py

## Contribuindo
Contribuições são bem-vindas. 

## Autores
Matheus Bonjour Laviola da Silva - Desenvolvimento inicial (dissertação de mestrado)


