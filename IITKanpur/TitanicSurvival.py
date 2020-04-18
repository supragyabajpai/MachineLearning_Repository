Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@supragyabajpai 
supragyabajpai
/
MachineLearning_Repository
1
00
 Code Issues 0 Pull requests 0 Actions Projects 0 Wiki Security Insights Settings
MachineLearning_Repository
/
Titanic_Survival_prediction.py
 

318
​
319
​
320
​
321
​
322
​
323
​
324
print( train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]  )
325
​
326
​
327
​
328
​
329
print( "Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100  )
330
print( "Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100  )
331
​
332
​
333
​
334
#Percentage of females who survived: 74.2038216561
335
#Percentage of males who survived: 18.8908145581
336
​
337
​
338
#Some Observations from above output
339
#------------------------------------
340
# As predicted, females have a much higher chance of survival than males.
341
# The Sex feature is essential in our predictions.
342
​
343
​
344
​
345
​
346
​
347
​
348
#--------------------
349
#4.B) Pclass Feature
350
#--------------------
351
#draw a bar plot of survival by Pclass
352
sbn.barplot(x="Pclass", y="Survived", data=train)
353
plt.show()
354
​
355
​
356
#print( percentage of people by Pclass that survived
357
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)
358
​
359
print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)
360
​
361
print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
362
#Percentage of Pclass = 1 who survived: 62.962962963
363
#Percentage of Pclass = 2 who survived: 47.2826086957
364
#Percentage of Pclass = 3 who survived: 24.2362525458
365
​
366
print()
367
print( "Percentage of Pclass = 1 who survived:\n\n", train["Survived"][train["Pclass"] == 1].value_counts()  )
368
​
369
print()
370
print( "Percentage of Pclass = 1 who survived:\n\n", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)    )
371
​
372
print()
373
print( "Percentage of Pclass = 1 who survived:\n\n", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]     )
374
​
375
​
376
​
377
​
378
​
379
#Some Observations from above output
380
#------------------------------------
@supragyabajpai
Commit changes
Commit summary
Update Titanic_Survival_prediction.py
Optional extended description
Add an optional extended description…
 Commit directly to the master branch.
 Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
