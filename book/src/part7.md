# Part VII — Living With It

Two chapters, both about the gap between understanding a system and working with one.

**Chapter 21** is about seeing inside a running engine. Every metric it emits exists because
someone needed it to answer a question — and by this point in the book, you have already
asked most of those questions. The chapter reads the instrumentation as a map back to the
design, names the five numbers that actually tell you what is happening, and ends with a
tuning procedure ordered by which constraint each knob relieves.

**Chapter 22** is about changing it. The extension points are the architecture's seams, and
walking them is the last check that the earlier chapters landed: adding a model, adding a
kernel, adding an attention backend, porting to new hardware. Each is short to describe
precisely because the preceding chapters did the work of explaining what the interface is
protecting.

The chapter — and the book — ends with the project's own answer to a hard problem: how do
you test an inference engine when the property that matters most, *the model produces correct
output*, cannot be checked by unit tests?
