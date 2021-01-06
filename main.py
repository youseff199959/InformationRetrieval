from PositionalIndexBuilder import PositionalIndexBuilder
from VectorSpaceModelBuilder import VectorSpaceModelBuilder



positionalIndexBuilder = PositionalIndexBuilder("docs");

positionalIndexBuilder.buildPositionalIndex();
# print(positionalIndexBuilder.positionalIndex)
# print(positionalIndexBuilder.queryPhrase("new" , "post"))


vectorBuilder = VectorSpaceModelBuilder(positionalIndexBuilder.corpus , positionalIndexBuilder.corpusNormalizedAndTokenized , positionalIndexBuilder.positionalIndex)
print(vectorBuilder.findAllTermFreq())
print(vectorBuilder.findIDF())
print(vectorBuilder.getTFIDF() )



query = input('enter query or type cancel ')
while query.lower() != 'cancel':
    print(vectorBuilder.findSimilarityBetweenQueryAndAllDocs(query));

    query = input('enter query or type cancel ')
