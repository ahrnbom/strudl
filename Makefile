build:
	docker build -t strudl .
test: build
	nvidia-docker run -ti strudl py.test -v
push: build
	docker tag strudl hakanardo/strudl:dev || docker tag -f strudl hakanardo/strudl:dev
	docker push hakanardo/strudl:dev
singularity: push
	singularity build strudl.img docker://hakanardo/strudl:dev