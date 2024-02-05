rawcontent = r"""\begin{figure} 
    \centering
    \begin{subfigure}[b]{0.3\textwidth}
    \includegraphics[width=\textwidth]{fig/method_name_size_norm_0.png}
    \caption{0}
    \label{fig:image1}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.3\textwidth}
    \includegraphics[width=\textwidth]{fig/method_name_size_norm_20.png}
    \caption{20}
    \label{fig:image2}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.3\textwidth}
    \includegraphics[width=\textwidth]{fig/method_name_size_norm_90.png}
    \caption{90}
    \label{fig:image3}
  \end{subfigure}

  \caption{method,name,size,norm sequence}
  \label{fig:three_images}
\end{figure}"""


    # Repeat the content 10 times
for method in ["gibbs" , "pde"]:
    for norm in ["L1", "L2"]:
        for size in ["big","small"]:
            for name in ["stone", "sce", "room"]:
                content = rawcontent.replace(r"method", method)
                content = content.replace(r"norm", norm)
                content = content.replace(r"size",size)
                content = content.replace(r"name",name)
                with open("output.tex", "a") as file:
                    file.write(content)
    