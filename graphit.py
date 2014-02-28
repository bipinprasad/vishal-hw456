'''
Created on Feb 17, 2014

@author: Vishal
'''
import traceback

try:
    from pyx import graph, text, style, color
    text.set(texdebug="hello", errordebug=2)    # debug pyx
except:
    traceback.print_exc()
    print '-------------------------------------------------------------'
    print 'Graphing module PyX is not available.'
    print 'Please install (pip install PyX and and then run this program'
    print '-------------------------------------------------------------'


def plot2d(data, mytitle, xaxistitle="X", yaxistitle="Y", mycolor=(0,0,0), minpoint=None, maxpoint=None):
    '''
    data: list of tuples of (x,y)
    mytitle: determines the name of the of PDF file.
    
    Graphs min/max is automatically chosen as to cover the range with 0,1 being covered.
    To override, provide the defaults minpoint (x,y) and maxpoint(x,y)
    '''
    try:
        fnameprefix = '-'.join(mytitle.split()) + '-2d'
        datafile    = fnameprefix + '.dat'
        with open(datafile, 'w') as f:
            f.write('\n'.join(['%s\t%s' % (x,y) for x,y in data]))
            f.close()
            print 'Data File written: %s' % datafile
    except:
        print '%s: Failed to save data file' % (mytitle, datafile)
        traceback.print_exc()
    
    try:
        if minpoint:
            xmin,ymin = minpoint 
        else:
            xmin = min(0,min([x for x,y in data])) 
            ymin = min(0,min([y for x,y in data]))
        if maxpoint:
            xmax,ymax = maxpoint
        else:
            xmax = max(1,max([x for x,y in data]))
            ymax = max(1,max([y for x,y in data]))
            
        print 'xmin=%s, xmax=%s' % (xmin, xmax)
        print 'ymin=%s, ymax=%s' % (ymin, ymax)
        g = graph.graphxy(width=8,
                          x = graph.axis.linear(min=xmin, max=xmax,title=xaxistitle),
                          y = graph.axis.linear(min=ymin, max=ymax,title=yaxistitle))
        g.plot(graph.data.points(data, x=1,y=2),
               styles = [ graph.style.line([ style.linestyle.solid]) ] )
        g.text(g.width/2, g.height + 0.2, mytitle, [text.halign.center, text.valign.bottom,text.size.large])
        g.writePDFfile(fnameprefix)
        print 'Graph PDF File written: %s.pdf' % fnameprefix
    except:
        print '%s: Failed to create Graph' % mytitle
        traceback.print_exc()
        
def plotIterationInfo(transitionIterations, title):
    '''
    Do we have a p01 graph or a p10 graph. Figure it out and graph it.
    '''
    x,y,z = transitionIterations[0]
    if all([p01==x for p01,p10,prob in transitionIterations]):
        is_p01 = False
        data = [(p10,prob) for p01,p10,prob in transitionIterations]
    else:
        is_p01 = True
        data = [(p01,prob) for p01,p10,prob in transitionIterations]
        
    if is_p01:
        xaxistitle = 'Transition Probability From State 0 to 1'
    else:
        xaxistitle = 'Transition Probability From State 1 to 0'
    yaxistitle = 'Total Emission Probability'
    
    plot2d(data, title, xaxistitle=xaxistitle, yaxistitle=yaxistitle, mycolor=(0,0,0), minpoint=None, maxpoint=None)
        
def plotIteration3D(data, mytitle, xaxistitle="State Transition From 0 to 1", yaxistitle="State Transition From 0 to 1", mycolor=(0,0,0), minpoint=None, maxpoint=None):       
    # Save the data into file
    try:
        fnameprefix = '-'.join(mytitle.split()) + '-3d'
        datafile    = fnameprefix + '.dat'
        with open(datafile, 'w') as f:
            f.write('\n'.join(['%s\t%s\t%s' % (x,y,z) for x,y,z in data]))
            f.close()
            print 'Data File written: %s' % datafile
    except:
        print '%s: Failed to save data file' % (mytitle, datafile)
        traceback.print_exc()
    
    try:
        g = graph.graphxyz(size=4, projector=graph.graphxyz.parallel(170, 45))
        # g.plot(graph.data.file("color.dat", x=1, y=2, z=3, color=3),
        g.plot(graph.data.points(data, x=1, y=2, z=3, color=3),
               [graph.style.surface(gradient=color.gradient.RedGreen,
                                    gridcolor=color.rgb.black,
                                    backcolor=color.rgb.black)])
        g.writePDFfile(fnameprefix)
        print 'Graph PDF File written: %s.pdf' % fnameprefix
    except:
        print '%s: Failed to create Graph' % mytitle
        traceback.print_exc()
   

def main():
    '''
    Test code
    '''
    print 'Passed import'
    data = [(1,1), (1.5, 1.7), (2.0, 3.0), (2.5, 2.5), (3.0, 2.0), (3.5, 1.5), (4.0,0.5), (4.5,0)]
    plot2d(data, 'Test-Data')
    
if __name__ == '__main__':
    main()