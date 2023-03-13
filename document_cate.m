%----------------------------------------------------------------------------------Loading_data
dataset_name_csv = "dataset_20new.csv";                                    %path to dataset.csv file
dataset_20news = readtable(dataset_name_csv,'TextType','string');          %load csv to table

%----------------------------------------------------------------------------------Preprocessing
dataset_20news.group = categorical(dataset_20news.group);                  %extracting categories from column 'group'

cvp = cvpartition(dataset_20news.group,'Holdout',0.2);                     %split dataset on test and training (0.25-sets ratio)
train_data = dataset_20news(training(cvp),:);                              %define X_train
test_data = dataset_20news(test(cvp),:);                                   %define y_train

train_data_text = train_data.text;                                         %extracting text from data
test_data_text = test_data.text;                                           %

y_train = train_data.group;
y_test = test_data.group;


train_data_text = tokenizedDocument(train_data_text);                      %data tekonization
train_data_text = lower(train_data_text);                                  %data lowercase
train_data_text = erasePunctuation(train_data_text);                       %data erasing punctuation and symbols

test_data_text = tokenizedDocument(test_data_text);                        
test_data_text = lower(test_data_text);                                    
test_data_text = erasePunctuation(test_data_text);      


enc = wordEncoding(train_data_text);                                       %encoding text
sequence_length = 40;
X_Train = doc2sequence(enc,train_data_text,'Length',sequence_length);      %fitting enumerated text to set
X_Test = doc2sequence(enc,test_data_text,'Length',sequence_length);

%-------------------------------------------------------------------------------Model_1DConv
input_size = 1;
embedding_dimension = 100;

words_num = enc.NumWords;
classes_num = numel(categories(y_train));

layers = [                                                                 %model layers
    sequenceInputLayer(input_size)
    wordEmbeddingLayer(embedding_dimension,words_num)
    convolution1dLayer(2, 200, Padding="causal")
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    globalAveragePooling1dLayer
    fullyConnectedLayer(classes_num)
    softmaxLayer
    classificationLayer
]

options = trainingOptions("adam", ...
    MiniBatchSize=128, ...
    SequencePaddingDirection="left", ...
    ValidationData={X_Test,y_test}, ...
    OutputNetwork="best-validation-loss", ...
    Plots="training-progress", ...
    Verbose=0);

CNN = trainNetwork(X_Train,y_train,layers,options);                        %training


%-----------------------------------------------------------------------------------New_predictions
new_doc = ['<Most women right off the bat want to know how tall the motorcycle ' ...
    'is and if they can put their feet flat on the ground. The seat height on the' ...
    ' new Softail Standard is 25.8 inches, which is one of the lowest in Harley-Davidson’s ' ...
    'lineup. I can attest, having ridden hundreds of motorcycles, that this is a manageable seat' ...
    ' height for most riders of average height.And judging by the pictures below, the woman riding' ...
    ' the Softail Standard appears to be quite comfortable on the motorcycle. Until I can test ride one, ' ...
    'I can only judge the comfort of the riding position by the photos supplied by Harley-Davidson. Here’' ...
    's my assessment, again without actually riding one. I highly recommend visiting a dealership as soon ' ...
    'as you are able and sitting on or test riding one to get a good feel for how you actually fit on the' ...
    ' motorcycle.The Softail Standard features all the same powertrain elements that are on most of the' ...
    ' other Softails including the Milwaukee-Eight 107 V-Twin engine (1746cc), with its counter-balancers ' ...
    'that reduce vibration at idle speeds to improve rider comfort (essentially rubber pads that cushion ' ...
    'the engine where its mounted to the frame). The Softail Standard is equipped with a 3.5-gallon fuel ' ...
    'tank and averages 47 mph, providing plenty of juice to keep you riding before your body needs a break.' ...
    'If you want more color options than black and a little more rebel styling, spend another $1,000 for the' ...
    ' Street Bob and you’ll get essentially the same motorcycle, but with additional custom touches like blacked ' ...
    'out and brushed-black chrome pieces. But if you’re like most riders, you likely can’t resist changing out' ...
    ' parts on the motorcycle, so my recommendation is to get the lower priced Standard and switch out the parts ' ...
    'from there.The bobber style of motorcycle is still very much in demand among cruiser riders as well as folks ' ...
    'getting into motorcycling for the first time. To capitalize on the interest, Harley-Davidson is rolling out' ...
    ' a new model for 2020, the Softail Standard, and pricing it at an attainable entry point for a Harley-Davidson Big Twin at $13,599.The Softail Standard is designed as a “minimalist” motorcycle encompassing elements that attract riders to the classically cool bobber style, including a shorter rear fender, mini-ape handlebars, solo seat, and an upright seating position with feet right below the knees.>']

new_doc = tokenizedDocument(new_doc);                        %new_doc preprocessing                     
new_doc = lower(new_doc);                                    
new_doc = erasePunctuation(new_doc);  

X_new = doc2sequence(enc,new_doc,'Length',sequence_length);

new_labels = classify(CNN,X_new)

%new_doc = tokenizedDocument(new_doc);new_doc = lower(new_doc);new_doc = erasePunctuation(new_doc);X_new = doc2sequence(enc,new_doc,'Length',sequence_length);new_labels = classify(CNN,X_new)

%-----------------------------------------------------------------------------------Export
%filename = "doc_cat.onnx"
%exportONNXNetwork(net,filename)
