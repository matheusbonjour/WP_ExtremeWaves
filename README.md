# Clusterização de Padrões Atmosféricos associados a Extremos de Onda

Este repositório contém um conjunto de scripts Python desenvolvidos para identificar e analisar padrões atmosféricos associados a extremos de onda ao longo da costa sul e sudeste do Brasil. O trabalho baseia-se no uso do K-Means para clusterização dos padrões atmosféricos associados a extremos de onda. 

O Artigo publicado associado a esse algortimo encontra-se em:  https://link.springer.com/article/10.1007/s11069-025-07166-7

Os dados para teste estão dispoíveis em: https://drive.google.com/drive/folders/1hejo_AZWd4qkTQNJyEvG2YT9sy4-XVKm?hl=pt-br

## Estrutura do Repositório

Os scripts estão organizados no diretório `src` e cada um possui uma funcionalidade específica dentro do processo de clusterização:

- `boia_domain_download.py`: Responsável por baixar dados de reanálise atmosférica e dados de ondas de boias específicas.
- `GetDays.py`: Seleciona os dias com eventos extremos de ondas com base em um limiar predefinido (por exemplo, o percentil 99).
- `KneeLocWP.py`: Determina o número ótimo de clusters para o algoritmo K-Means usando o método do cotovelo.
- `mainWP.py`: Script principal que coordena o processo de clusterização, chamando as funções necessárias dos outros scripts.
- `PlotWP.py`: Gera visualizações dos padrões atmosféricos identificados, como mapas de clusters e possivelmente séries temporais dos eventos.
- `ProcessWP.py`:

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


