# ECB+ Image Annotations

## Annotation Process
For each ECB+ document (all of which have associated URLs), the associated URL was visited. If the link produced an error, redirected to a home page or other article, or domain name was no longer valid, [web.archive.org](web.archive.org) was checked for an archived copy of the page. Likewise, if a functional page had embedded images that were no longer hosted, the image URLs were checked against the Internet Archive.
None of the ECB documents had associated URLs, and so were attempted to be located by a combination of broad and exact match Google searches of the document contents.

## File Structure
Each subfolder in this directory corresponds to one ECB+ (including original ECB) document, where the first number in the file name corresponds to the ECB+ topic number, and the second to the number of the article within that topic. "ecbplus" folders may contain a "broken_url.txt" file, indicating that the associated URL was no longer functional; this file may contain a URL to a web archive of the original page, if one existed. "ecb" folders may contain a "found_url.txt" which contains a recovered URL for the document. Image captions are stored in .txt files with the same file name (save for the ending) as the image they correspond to, e.g. `lindsay court 661 reuters.txt` is the caption for `lindsay court 661 reuters.jpg`.

## Summary
* Of the 502 ECB+ links, 214 were broken (~43%). Of those, 106 (~50%) were able to be recovered using [web.archive.org](web.archive.org)
* Of the 480 ECB document, 246 (~51%) were able to be located via Google search.
* In total 543 images were retrieved, 332 (~61%) of which had an associated caption. Avg. ~13 images per topic / ~0.55 per article.

Unique image file endings:
```
['.jpg', '.webp', '.jfif', '.jpeg', '.png', '.gif', '.avif', '.JPG', '.PNG']
```
(These are now standardized to .jpg)
