TAG=12
all:
	docker build -t matei10/pyfaas_scale:$(TAG) -f Dockerfile.scale .
	docker build -t matei10/pyfaas_predict:$(TAG) -f Dockerfile.predict .
	docker push matei10/pyfaas_predict:$(TAG)
	docker push matei10/pyfaas_scale:$(TAG)

test:
	 python3 -m pytest --cov=scale .
	 pylint scale.py
