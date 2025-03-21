# Insect Species Identification

## Android app APK

[Download APK file](https://drive.google.com/drive/folders/1UNogisKp3rtcOnigcibAPiNsQB-gZJpD?usp=drive_link)

## Android app screenshots

<p align="center">
	<img src="insect-id-app/screenshots/1.jpg" alt="Screenshot" width="125"/>
	<img src="insect-id-app/screenshots/2.jpg" alt="Screenshot" width="125"/>
	<img src="insect-id-app/screenshots/3.jpg" alt="Screenshot" width="125"/>
	<img src="insect-id-app/screenshots/4.jpg" alt="Screenshot" width="125"/>
	<img src="insect-id-app/screenshots/5.jpg" alt="Screenshot" width="125"/>
	<img src="insect-id-app/screenshots/6.jpg" alt="Screenshot" width="125"/>
</p>

## List of species and classes trained on

[All species](https://github.com/rakeshmalik91/insect-id/blob/main/species.json) (contains species that do not have any images available as well)

- [Lepidoptera classes](https://github.com/rakeshmalik91/insect-id/blob/main/models/classes.lepidoptera.json)
	- [Butterfly classes](https://github.com/rakeshmalik91/insect-id/blob/main/models/classes.butterfly.json)
	- [Moth classes](https://github.com/rakeshmalik91/insect-id/blob/main/models/classes.moth.json)
- [Odonata (Dragonfly/Damselfly) classes](https://github.com/rakeshmalik91/insect-id/blob/main/models/classes.odonata.json)
- [Cicada classes](https://github.com/rakeshmalik91/insect-id/blob/main/models/classes.cicada.json)

Note: early stage classes suffixed with "-early"

Note: spp. classes suffixed with "-spp" or "-genera" or "-genera-spp"

## Android app dependencies

- [OpenCV 4.11.0](https://github.com/opencv/opencv/releases/tag/4.11.0) (location: insect-id-app/opencv)

## Model checkpoints

[Model checkpoints](https://drive.google.com/drive/folders/1FtGjLJc_JNwLs0cey3euyzUxwpids10G?usp=drive_link)

## Datasets trained on

[Datasets](https://drive.google.com/drive/folders/10qLVcGkJlLplKjIluRc9GEyQhcqpyhhD?usp=drive_link)

| Source					| Image count | Class count | Imago class | Early stage class | Species type   | Region   | Comments
|---------------------------|-------------|-------------|-------------|-------------------|----------------|----------|------------------------------------
| mothsofindia.org   		| 44k         | 3364        | 3060        | 304               | moth           | india    | Contains 411 spp. classes
| ifoundbutterflies.org   	| 66k         | 1554        | 1125        | 429               | butterfly      | india    | Contains 35 spp. classes
| indianodonata.org			| 13k         | 737         | 510         | 200               | odonata        | india    | Contains 27 spp. classes <br/>& 157 empty classes
| indiancicada.org		   	| 1k          | 308         | 308         | 7                 | cicada         | india    | Contains 1 spp. classes <br/>& 139 empty classes
| inaturalist.org           | 232k        | 4221        | 3773        | 448               | all            | india    | Contains <br/>2732 moth, <br/>976 butterfly, <br/>370 odonata, <br/>154 cicada classes
| indiabiodiversity.org   	| 12k         | 1444        | 1444        | 0                 | moth+butterfly | india    | Contains typo in class names, <br/>uses legacy class names
| insecta.pro               | 25k         | 5068        | 5068        | 0                 | moth+butterfly | all      | Low res images (320x~250)
| wikipedia.org				| 2k          | 1825        | 1825        | 0                 | moth+butterfly | india    | Low res images (220x~160)
