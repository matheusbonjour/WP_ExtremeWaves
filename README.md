# Clusterização de Padrões Atmosféricos e Extremos de Onda

Este repositório contém um conjunto de scripts Python desenvolvidos para identificar e analisar padrões atmosféricos associados a extremos de onda ao longo da costa sul e sudeste do Brasil. O trabalho baseia-se em técnicas de aprendizado de máquina para processar e analisar dados atmosféricos e oceânicos.

## Estrutura do Repositório

Os scripts estão organizados no diretório `src` e cada um possui uma funcionalidade específica dentro do processo de clusterização:

- `boia_domain_download.py`: Responsável por baixar dados de reanálise atmosférica e dados de ondas de boias específicas.
- `GetDays.py`: Seleciona os dias com eventos extremos de ondas com base em um limiar predefinido (por exemplo, o percentil 99).
- `KneeLocWP.py`: Determina o número ótimo de clusters para o algoritmo K-Means usando o método do cotovelo.
- `mainWP.py`: Script principal que coordena o processo de clusterização, chamando as funções necessárias dos outros scripts.
- `PlotWP.py`: Gera visualizações dos padrões atmosféricos identificados, como mapas de clusters e possivelmente séries temporais dos eventos.
- `ProcessWP.py`:

## Instalação

Para instalar as dependências necessárias, execute o seguinte comando:

numpy
pandas
xarray
scikit-learn
matplotlib

## Uso
Para executar a análise completa, inicie o mainWP.py da seguinte forma:

python src/mainWP.py

## Contribuindo
Contribuições para este projeto são bem-vindas. Veja CONTRIBUTING.md para mais detalhes sobre como contribuir.

## Autores
Matheus Bonjour Laviola da Silva - Desenvolvimento inicial

## Licença
Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

