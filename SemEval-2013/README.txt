==========================================================
SemEval-2013 - Word Sense Induction & Disambiguation within an End-User Application
Dataset
==========================================================

SemEval-2013 - Task #11  site: http://www.cs.york.ac.uk/semeval-2013/task11/

==========================================================

This dataset is designed for evaluation of subtopic information retrieval.
For details, please refer to our paper:

A. Di Marco, R. Navigli. Clustering and Diversifying Web Search Results
with Graph-Based Word Sense Induction.Computational Linguistics, 39(4), MIT Press, 2013

CONTENTS

This package contains the dataset which consists of 100 topics (all nouns),
each with a list of 64 top-ranking documents.

The topics were selected from the list of ambiguous Wikipedia entries;
i.e., those with "disambiguation" in the title (see
http://en.wikipedia.org/wiki/Wikipedia:Links_to_%28disambiguation%29_pages)

It includes queries of length ranged between 1 and 4 words.

length    queries
 1         40
 2         40
 3         15
 4          5

The 64 snippets associated with each topic were collected from the Google
search engine from the last months of 2012, and they were subsequently annotated with
subtopic relevance judgments.

Files:

 README.txt	This file.
 2010/		This folder contains 100 xml files.
 topics.txt	contains the topics.
 results.txt	contains the search engine ranking of the results.
 subTopics.txt	contains the Wikipedia subTopics of the queries.
 STRel.txt	contains the gold annotations.

==========================================================
==========================================================

#Format:
==================== FORMAT 2010 =========================
(SemEval-2010 Word Sense Induction & Disambiguation Task #14)

The dataset in the format 2010 consists of 100 files xml, one for each query.


Example of a xml file:

==================== apple.n.xml ========================
            
<apple.n.23 url="http://trailers.apple.com/trailers/" title="Apple Trailers - iTunes Movie Trailers">
   <TargetSentence>
      View the latest movie trailers for many current and upcoming releases.
      Many trailers are available in high-quality HD, iPod, and iPhone versions.
   </TargetSentence>
</apple.n.23>

==================== FORMAT 2013 =========================

The dataset in the format 2013 consists of two files where each row is terminated by
Linefeed (ASCII 10) and fields are separated by Tab (ASCII 9).
The two files are described below:

==================== topics.txt =========================

It contains topic id and description:

id    description
38    shockwave
39    apache
40    magic
.........

==================== results.txt ========================

It contains result ID (formed by topic ID and search engine rank of
result), URL,  title, and snippet:

ID    url    title    snippet
39.1    http://www.apache.org/    Welcome to The Apache Software Foundation!    Supports the development of a number of open-source software projects,   including the <b>Apache</b> web server. Includes license information, latest news, and   project <b>...</b>
39.2    http://httpd.apache.org/    Welcome! - The Apache HTTP Server Project    The <b>Apache</b> HTTP Server Project is an effort to develop and maintain an open-  source HTTP server for modern operating systems including UNIX and Windows <b>...</b>
39.3    http://www.apachecorp.com/    Apache Corporation : Home    <b>Apache</b> Corporation is an oil and gas exploration and production company with   operations in the United States, Canada, Egypt, the United Kingdom North Sea, <b>...</b>
........

==========================================================
==================== AUTHORS =============================

Roberto Navigli, Sapienza University of Rome
(navigli@di.uniroma1.it)

Daniele Vannella, Sapienza University of Rome
(vannella@di.uniroma1.it)

==================== CONTACT =============================

Please feel free to get in touch with us for any questions or problems you may have:

 http://groups.google.com/group/semeval-2013-wsi-in-application?hl=en

=============== COPYRIGHT AND LICENSE ====================

This work is licensed under the Creative Commons
Attribution-Noncommercial-Share Alike 3.0 Unported License. To view a
copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/

================ ACKNOWLEDGMENTS =========================

The authors gratefully acknowledge the support of the MultiJEDI ERC Starting Grant No. 259234
(http://lcl.uniroma1.it/multijedi) and the CASPUR High-Performance Computing Grant 475/2011.