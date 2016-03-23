# And Then There Were None
Dementia affects more than one third of adults over 85 years old and one in nine adults over 65.  Alzheimer's Association estimates that half of people with Alzheimer's do not know they have it.  It has been shown that Agatha Christie used fewer unique words and more indefinite words in her later novels.  I wanted to take this research further by uncovering other features and creating an early warning system.  And Then There Were None is a prototype for detecting dementia through someone's writing.  

##  Model Summary
Features from various authors' novels were created using Natural Language Processing.  Parts of speech, syntactic structure, n-grams, and topic clustering were all examined.  The features with the most signal were input into a Gausssian KDE distribution and fit based on an author's earlier works.  Authors with dementia had their works post dementia onset slowly drift away from their fitted distribution, indicating a fundamental change in their writing.

## Visual Representation of Model
This is a contour plot of the first two principal components fit to a Gaussian KDE.  Each point represents a novel and is annotated with the author's age.  One possible interpretation of the plot is that the further from the boundary the points are, the less the author resembles his/her previous self.  Drops appear in chronological order.

#### Agatha Christie - Had Dementia
![Agatha Christie](img/Agatha_Christie.gif "Agatha Christie")

#### Ross MacDonald - Had Dementia
![Ross MacDonald](img/Ross_MacDonald.gif "Ross MacDonald")

#### Stephen King - No Known Dementia
![Stephen King](img/Stephen_King.gif "Stephen King")

## Data Analysis
Interesting features that stood out:
##### Coming Soon


## Technologies Used
* Python
* Numpy
* Pandas
* PostgreSQL
* Sklearn
* NLTK
* SpaCy

