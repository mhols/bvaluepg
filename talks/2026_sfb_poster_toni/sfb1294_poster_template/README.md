# CRC 1294 poster template

Provides a unified template for CRC posters which should be used at official CRC
meetings. May also be used for presentations of CRC content at other occasions.
Contains CRC and institutional logo, references and box for the (mandatory)
acknowledgment.

## Edit file

Edit [Poster_SR.tex](Poster_SR.tex) and adapt language, institution, project id,
title, authors, references and acknowledgments as needed.

## Build template

*tl;dr Use ```latex+dvips+ps2pdf``` to build your poster as PDF. Do* ***not***
*use ```pdflatex``` as this will fail!*

The poster LaTeX class does not allow for a direct use of latexpdf. Instead
please use

```
latex YOUR_FILE.tex
```

to produce a file called ```YOUR_FILE.dvi```. This in turn can be used to
produce a postscript file via

```
dvips -o YOUR_FILE.ps YOUR_FILE
```

Most PDF viewers will then automatically turn ```YOUR_FILE.ps``` into a PDF
file. If not, use

```
ps2pdf YOUR_FILE.ps
```

to create a ```YOUR_FILE.pdf```.

In order for this to work, your images should all be included as postscript files (.eps) using
the command

```
\includegraphics[]{}
```


## Work with overleaf or similar

The default compiler in overleaf is ```pdflatex```.
Do* ***not*** *use ```pdflatex``` as this will fail!*
Instead got to the project menu and change the compiler to ```LaTeX```