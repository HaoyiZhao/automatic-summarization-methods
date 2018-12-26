# automatic-summarization-methods
Several techniques used for automatic summarization of articles. Methods consist of SumBasic, and derivations/extensions of SumBasic.

Command structure: <br />
python sumbasic.py <method_name> <file_n>*

Output summary is written to standard output.

e.g:<br />
python ./sumbasic.py simplified ./docs/doc1-*.txt > simplified-1.txt

# Example documents
Example documents to demonstrate use of the methods are included in the docs folder. The documents consist of news articles that are clustered based on topics. doc1-*.txt represents the first topic cluster, doc2-*.txt represents the second, etc.
