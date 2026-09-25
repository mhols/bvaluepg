%
% Version: Vers.1 sfb1294_1-poster.cls, 2017/01/12
%
% LaTeX class sfb1294_1-poster
%
% for the CRC 1294
% Data Assimilation
%
% ----------------------------------------------------------------------
%
%
% ----------------------------------------------------------------------
% The options for this class are:
% + german  - select german title
% + english - select english title (default)
% + E  - no non-UP PI
% + HU      - select HU as the institution
% + TU      - select TU as the institution
% + WIAS    - select Wias as the institution
% + GFZ		- select GFZ as the institution
% + TUI		- select TU Ilmenau as the institution
% + reArea  - selects the title for a research area (appl)
% ----------------------------------------------------------------------
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sfb1294_1-poster}%[2017/01/05 v3.1]
%
\RequirePackage{ifthen}
%
% ----------------------------------------------------------------------
% ---------------------- Handle Options --------------------------------
% ----------------------------------------------------------------------
% @germanlanguage stores whether we use german as the language
% for titles etc.
\newboolean{@germanlanguage}
\setboolean{@germanlanguage}{false}
%
\DeclareOption{german}{%
  \setboolean{@germanlanguage}{true}%
}
\DeclareOption{english}{}  % do nothing since default is english
%
% Define option for application area posters
\newboolean{@researchareaposter}
\setboolean{@researchareaposter}{false}
%
\DeclareOption{reArea}{%
  \setboolean{@researchareaposter}{true}%
}
%
% We keep track of the number of institutions:
\newcounter{@numinst}
\setcounter{@numinst}{0}
% Declare options for each institution
\DeclareOption{E}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{%
	\newcommand{\@insttext}{}%
    \newcommand{\@instlogo}{}%
	}{%
	\ifthenelse{\value{@numinst}=2}{%
		\newcommand{\@insttextTwo}{}%
		\newcommand{\@instlogoTwo}{}%
	}{}%
 }%
}%
%
%
\DeclareOption{UP}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{% 
    \newcommand{\@insttext}{Universit\"at Potsdam\\\@matheonurl{www.uni-potsdam.de}}%
    \newcommand{\@instlogo}{\includegraphics[width=0.4\linewidth]{up_logo-weiss}}%
  }{%
    \ifthenelse{\value{@numinst}=2}{%
      \newcommand{\@insttextTwo}{Universit\"at Potsdam\\\@matheonurl{www.uni-potsdam.de}}%
      \newcommand{\@instlogoTwo}{\includegraphics[width=0.4\linewidth]{up_logo-weiss}}%
  	}{}%
  }%
}%
%
\DeclareOption{TU}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{%
    \newcommand{\@insttext}{Technische Universit\"at Berlin\\\@matheonurl{www.tu-berlin.de}}%
    \newcommand{\@instlogo}{\includegraphics[width=0.5\linewidth]{tu_logo-weiss}}%
  }{%
    \ifthenelse{\value{@numinst}=2}{%
      \newcommand{\@insttextTwo}{Technische Universit\"at Berlin\\\@matheonurl{www.tu-berlin.de}}%
      \newcommand{\@instlogoTwo}{\includegraphics[width=0.5\linewidth]{tu_logo-weiss}}%
    }{}%
  }%
}%
%
\DeclareOption{HU}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{%
    \newcommand{\@insttext}{Humboldt Universit\"at Berlin\\\@matheonurl{www.hu-berlin.de}}%
    \newcommand{\@instlogo}{\includegraphics[width=0.5\linewidth]{husiegel_bw}}%
  }{%
    \ifthenelse{\value{@numinst}=2}{%
      \newcommand{\@insttextTwo}{Humboldt Universit\"at Berlin\\\@matheonurl{www.hu-berlin.de}}%
      \newcommand{\@instlogoTwo}{\includegraphics[width=0.5\linewidth]{husiegel_bw}}%
    }{}%
  }%
}%
%
%
\DeclareOption{GFZ}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{%
    \newcommand{\@insttext}{Deutsches
		\\ GeoForschungsZentrum\\%
      \@matheonurl{www.gfz-potsdam.de}}%
    \newcommand{\@instlogo}{\includegraphics[width=0.7\linewidth]{GFZ-CD_LogoCMYK_DE_oRahmen-weiss.eps}}%
  }{%
    \ifthenelse{\value{@numinst}=2}{%
      \newcommand{\@insttextTwo}{Deutsches
		\\ GeoForschungsZentrum\\%
        \@matheonurl{www.gfz-potsdam.de}}%
      \newcommand{\@instlogoTwo}{\includegraphics[width=0.7\linewidth]{GFZ-CD_LogoCMYK_DE_oRahmen-weiss.eps}}%
    }{}%
  }%
}

\DeclareOption{TUI}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{%
    \newcommand{\@insttext}{Technische Universit\"at 
		\\ Ilmenau\\%
      \@matheonurl{www.tu-ilmenau.de}}%
    \newcommand{\@instlogo}{\includegraphics[width=0.7\linewidth]{TU_Ilmenau_Logo_black_green.eps}}%
  }{%
    \ifthenelse{\value{@numinst}=2}{%
      \newcommand{\@insttextTwo}{Technische Universit\"at
		\\ Ilmenau\\%
        \@matheonurl{www.tu-ilmenau.de/}}%
      \newcommand{\@instlogoTwo}{\includegraphics[width=0.7\linewidth]{TU_Ilmenau_Logo_black_green.eps}}%
    }{}%
  }%
}

\DeclareOption{URO}{%
  \stepcounter{@numinst}%
  \ifthenelse{\value{@numinst}=1}{%
    \newcommand{\@insttext}{Universit\"at Rostock\\%
      \@matheonurl{https://www.uni-rostock.de/}}%
    \newcommand{\@instlogo}{\includegraphics[width=0.7\linewidth]{Logo_weiss_Rostock_20x8cm.eps}}%
  }{%
    \ifthenelse{\value{@numinst}=2}{%
      \newcommand{\@insttextTwo}{Universit\"at Rostock\\%
        \@matheonurl{https://www.uni-rostock.de/}}%
      \newcommand{\@instlogoTwo}{\includegraphics[width=0.7\linewidth]{Logo_weiss_Rostock_20x8cm.eps}}%
    }{}%
  }%
}

%
\ProcessOptions*\relax
%
% check whether the combination of options is valid
%
\newboolean{@twoinst}
\setboolean{@twoinst}{false}
\ifthenelse{\boolean{@researchareaposter}}
{
  \ifthenelse{\the@numinst>0}{%
    \ClassError{sfb1294_1-poster}{Selected research area poster, but supplied an institution!}{}%
  }{}%
}
{
  \ifthenelse{\value{@numinst}=0}{%
    \ClassError{sfb1294_1-poster}{No institution selected!}{Please choose UP,
      HU, TU, WIAS, TUI, or GFZ as an option to the class sfb1294_1-poster.}
  }{%
    \ifthenelse{\value{@numinst}=2}{\setboolean{@twoinst}{true}}{%
      \ifthenelse{\value{@numinst}>2}{%
        \ClassError{sfb1294_1-poster}{This class can only handle at most two external instiutions!}{}%
      }{}%
    }%
  }%
}
%
% ----------------------------------------------------------------------
% ---------------------- Load Packages ---------------------------------
% ----------------------------------------------------------------------
%
% The following is based on the article class.  Maybe this dependence
% should be removed, but the article class defines many usefull commands
% that we otherwise would have to define newly.
\LoadClass[10pt,a4paper]{article}
%
\if@germanlanguage
  \RequirePackage[ngerman]{babel}
\fi
%
% DIN-ISO-convertion (for german umlaute)
\RequirePackage[latin1]{inputenc}
%
% New font encoding (for hyphenization with umlaute)
\RequirePackage[T1]{fontenc}
%
% Use scalable postscript helvetica fonts
% We use the scaling in order to make characters in math mode
% and text mode about equal in size.
\RequirePackage[scaled=0.92]{helvet}
%
% URLs
\RequirePackage{url}
%
% For calculations within this class
\RequirePackage{calc}
%
% For including graphics
\RequirePackage[dvips]{graphicx}
%
% For colored text, background, etc.
\RequirePackage[dvips]{xcolor}
%
% We use the geometry package to set the size of the text etc.
\RequirePackage[a4paper,totalwidth=\paperwidth,totalheight=\paperheight]{geometry}
%
% For including the background picture
\RequirePackage[dvips,pscoord]{eso-pic}
%
% Turn on helvetica font
\AtBeginDocument{\sffamily}
%
% ----------------------------------------------------------------------
% ---------------------- Other Definitions -----------------------------
% ----------------------------------------------------------------------
%
% Turn of indentation of paragraphs
\setlength{\parindent}{0mm}
% modify the left margins for itemize, enumerate, etc.
\setlength{\leftmargini}{1.1em}
\setlength{\leftmarginii}{1em}
\setlength{\leftmarginiii}{0.9em}
\setlength{\leftmarginiv}{0.9em}
\setlength{\labelsep}{.25em}
%
% turn off page numbers
\pagestyle{empty}
%
% Define usefull font commands
\newcommand{\@matheontextfont}{\fontfamily{phv}\fontseries{m}\fontshape{n}\fontsize{7pt}{9pt}\selectfont}
\newcommand{\@matheonwwwfont}{\fontfamily{phv}\fontseries{m}\fontshape{n}\fontsize{7pt}{9pt}\selectfont}
\newcommand{\@matheonurl}[1]{{\@matheonwwwfont\bfseries\url{#1}}}
%
% Define the macro for including a background picture:
\newcommand{\bgpic}[1]{%
  \AddToShipoutPicture{%
    \setlength{\unitlength}{1mm}%
    \put(0,0){%
      \parbox[b][\paperheight][b]{\paperwidth}{%
       \vfill\centering
       \includegraphics[width=\paperwidth,height=\paperheight]{#1}%
       \vfill
     }%
   }%
  }%
}
% Same macro, put allow to change dimensions
\newcommand{\bgpicD}[3]{%
  \AddToShipoutPicture{%
    \setlength{\unitlength}{1mm}%
    \put(0,0){%
      \parbox[b][\paperheight][b]{\paperwidth}{%
       \vfill\centering
       \includegraphics[width=#1,height=#2]{#3}%
     }%
   }%
  }%
}
%
% Define some useful lengths
%
% \@logotitleboxwidth is the width of the logo part in the title
\newlength{\@logotitleboxwidth}
\setlength{\@logotitleboxwidth}{0.20\textwidth}
% \@titleboxheight is the height of the title
\newlength{\@titleboxheight}
\setlength{\@titleboxheight}{3.5cm}
%
% \contentswidth and \contentsheight store the width and height,
% respectively, of the remaining space (i.e, without titlebox)
% on the poster.
\newlength{\contentswidth}
\setlength{\contentswidth}{\textwidth}
\addtolength{\contentswidth}{-1.25pt}      % subtract small margin on the right
%
\newlength{\contentsheight}
\setlength{\contentsheight}{\textheight}
\addtolength{\contentsheight}{-\@titleboxheight}  % subtract titleheight
\addtolength{\contentsheight}{-3pt}               % subtract white separator line
%
%
% ----------------------------------------------------------------------
% ---------------------- Color Definitions -----------------------------
% ----------------------------------------------------------------------
% UP colors
\definecolor{up-blue}{RGB}{0,48,94} % blue UP text
\definecolor{mathnat-blue}{RGB}{0,128,181} % blue Math-Nat Faculty UP
\definecolor{up-grey}{RGB}{193,211,224}
%
% set page color
\pagecolor{up-grey}
%
% Set tab colors
\definecolor{tabcolor}{named}{mathnat-blue}
\definecolor{tabtextcolor}{named}{white}
%
% other useful colors
% Einige Farbdefinitionen, inclusive der Farbdefinitionen aus XFig
\definecolor{gray2}{gray}{0.85}
\definecolor{dimgray}{gray}{0.65}
\definecolor{light-gray}{gray}{0.79}
\definecolor{dark-gray}{gray}{0.1}
%
\definecolor{cyan}{rgb}{0.000,1.000,1.000}
\definecolor{magenta}{rgb}{1.000,0.000,1.000}
%
\definecolor{blue4}{rgb}{0.000,0.000,0.560}
\definecolor{blue3}{rgb}{0.000,0.000,0.690}
\definecolor{blue2}{rgb}{0.000,0.000,0.820}
\definecolor{ltblue}{rgb}{0.530,0.810,1.000}
%
\definecolor{green4}{rgb}{0.000,0.560,0.000}
\definecolor{green3}{rgb}{0.000,0.690,0.000}
\definecolor{green2}{rgb}{0.000,0.820,0.000}
%
\definecolor{cyan4}{rgb}{0.000,0.560,0.560}
\definecolor{cyan3}{rgb}{0.000,0.690,0.690}
\definecolor{cyan2}{rgb}{0.000,0.820,0.820}
%
\definecolor{red4}{rgb}{0.560,0.000,0.000}
\definecolor{red3}{rgb}{0.690,0.000,0.000}
\definecolor{red2}{rgb}{0.820,0.000,0.000}
%
\definecolor{magenta4}{rgb}{0.560,0.000,0.560}
\definecolor{magenta3}{rgb}{0.690,0.000,0.690}
\definecolor{magenta2}{rgb}{0.820,0.000,0.820}
%
\definecolor{brown4}{rgb}{0.500,0.190,0.000}
\definecolor{brown3}{rgb}{0.630,0.250,0.000}
\definecolor{brown2}{rgb}{0.750,0.380,0.000}
%
\definecolor{pink4}{rgb}{1.000,0.500,0.500}
\definecolor{pink3}{rgb}{1.000,0.630,0.630}
\definecolor{pink2}{rgb}{1.000,0.750,0.750}
\definecolor{pink}{rgb}{1.000,0.880,0.880}
%
\definecolor{gold}{rgb}{1.000,0.840,0.000}
%
% ----------------------------------------------------------------------
% ------------- Setting the Title - Project Variant --------------------
% ----------------------------------------------------------------------
%
% Sets up the texts for the titlebox depending on the language
\newcommand{\@insttextUP}{SFB 1294 Data Assimilation\\\@matheonurl{www.sfb1294.de}}%
\newcommand{\@instlogoUP}{\includegraphics[width=0.9\linewidth]{sfb_Logo_negativ_Druck.eps}}%
%
%
% \@titlebox{ project name }{ title }{ list of persons }
% prints the titlebox of the poster
\newcommand{\@titlebox}[3]{%
  % prepare the sizes
  \newlength{\@texttitleboxwidth}%
  \setlength{\@texttitleboxwidth}{\textwidth}%
  % subtract space needed for the logos
  \ifthenelse{\boolean{@twoinst}}{%
    \addtolength{\@texttitleboxwidth}{-3\@logotitleboxwidth}}{%
    \addtolength{\@texttitleboxwidth}{-2\@logotitleboxwidth}}%
  % subtract space needed for the vertical rules
  \addtolength{\@texttitleboxwidth}{-4pt}%
  % ---------------------------------------------------------------
  \parbox[b]{\textwidth}{%
    \color{white}%
    % set SFB logo on the left
    \setlength{\fboxsep}{0ex}%
    \setlength{\fboxrule}{0pt}%
      \colorbox{up-blue}{%
      \parbox[t][\@titleboxheight][t]{\@logotitleboxwidth}{%
        % set institution logo
        \parbox[t][0.7\@titleboxheight][c]{\linewidth}{%
          \centering
          \@instlogoUP
        }\\
        % set text
        \parbox[t][0.25\@titleboxheight][c]{\linewidth}{%
          \centering
          \@matheontextfont
          \@insttextUP
        }%
      }%
    }%
    \vrule width 2pt%
    %
    % set text in the middle
    \colorbox{mathnat-blue}{%
      \parbox[t][\@titleboxheight][c]{\@texttitleboxwidth}{%
        \centering
        % set project number (if not empty)
        \ifthenelse{\equal{#1}{}}{%
          \parbox[t][0.15\@titleboxheight][c]{0.96\linewidth}{\small \mbox{}}
        }{%
          \ifthenelse{\boolean{@germanlanguage}}{%
            %\parbox[t][0.15\@titleboxheight][c]{0.96\linewidth}{\small \textbf{SFB 1294 Datenassimilation \hfill Projekt #1}}
          }{%
            %\parbox[t][0.15\@titleboxheight][c]{0.96\linewidth}{\small \textbf{CRC 1294 Data Assimilation \hfill Project #1}}
          }
        }
        % set project title
        \parbox[t][0.65\@titleboxheight][c]{0.96\linewidth}{%
          \ifthenelse{\value{@numinst}=2}{\Large}{\Large}
          \begin{flushleft}\textbf{#2}\end{flushleft}
        }
        % set persons
        \parbox[t][0.2\@titleboxheight][c]{0.96\linewidth}{%
          \normalsize%
          \begin{flushleft}#3\end{flushleft}%
        }%
      }%
    }%
    \vrule width 2pt%
    %
    % set logo(s) on the right
    \colorbox{up-blue}{%
      \parbox[t][\@titleboxheight][t]{\@logotitleboxwidth}{%
        % set institution logo
        \parbox[t][0.7\@titleboxheight][c]{\linewidth}{%
          \centering
          \@instlogo
        }\\
        % set text
        \parbox[t][0.25\@titleboxheight][c]{\linewidth}{%
          \centering
          \@matheontextfont
          \@insttext
        }%
      }%
    }%
    %
    % set second logo if necessary
    \ifthenelse{\boolean{@twoinst}}{%
      \colorbox{up-blue}{%
        \parbox[t][\@titleboxheight][t]{\@logotitleboxwidth}{%
          % set institution logo
          \parbox[t][0.7\@titleboxheight][c]{\linewidth}{%
            \centering
            \@instlogoTwo
          }\\
          % set text
          \parbox[t][0.25\@titleboxheight][c]{\linewidth}{%
            \centering
            \@matheontextfont
            \@insttextTwo
          }%
        }%
      }%
    }{}%
    \hrule height 2pt
  }
  \par
}
%
\ifthenelse{\boolean{@researchareaposter}}{%
  \newcommand{\titlebox}[3]{\@titleboxReArea{#1}{#2}{#3}}
}{%
  \newcommand{\titlebox}[3]{\@titlebox{#1}{#2}{#3}}
}
\endinput
%%
%% End of file `sfb1294_1-poster.sty'.
