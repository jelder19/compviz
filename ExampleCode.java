public class ExampleCode {

    public static void main(String[] args) {
        // Will probably want to use C++ List
        ArrayList<Person> people = new ArrayList<>();
        long startTime = System.currentTimeMillis();
        long totalTime = 5000;
        Mat frame;

        while (true) {
            getImage(frame);
            if (totalTime >= 5000) {
                detectPeople(people, frame);
                startTime = System.currentTimeMillis();
            } else {
                track(people, frame);
            }
            detectFaces(people,frame);
            displayPeople(people, frame);
            totalTime = System.currentTimeMillis() - startTime;
        }

    }

    private void detectPeople(ArrayList<Person> people, Mat frame) {
        ArrayList<Person> matched = new ArrayList<>();
        ArrayList<Rect> bodies = new ArrayList<>();
        hog(bodies);
        Iterator it1 = people.begin();
        while (it1 != people.end()) {
            Iterator it2 = bodies.begin();
            while (it2 != bodies.end()) {
                if ((*it1).bodyMatch(*it2)) {
                    matched.push_back(*it1);
                    bodies.erase(it2);
                    break;
                } else {
                    it2++;
                }
            }
            it1++;
        }

        for (Iterator it = bodies.begin(); it != bodies.end(); it++) {
            matched.push_back(*it);
        }
        people = matched;
    }

    private void detectFaces(ArrayList<Person> people, Mat frame) {
        for (Iterator it = people.begin(); it != people.end; it++) {
            if (!(*it).gotFace()) {
                if (violaJones(it,frame)) {
                    recognize(it,frame);
                }
            } else {
                if (!(*it).gotName()) {
                    recognize(it,frame);
                }
            }
        }
    }

    //Only perform VJ on the region of the frame in the body rectangle
    private boolean violaJones(Person guy, Mat frame);

    //Perform recognition on the area of the face Rect
    //Only set the name if confident in match, otherwise leave it Unknown
    private void recognize(Person guy, Mat frame);

    //Get all of the bodies detected in the frame
    private void hog(ArrayList<Rect> bodies, Mat frame);

    //Camshift stuff, track faces and bodies
    private void track(ArrayList<Person> people, Mat frame);

    //Display frame with bodies, faces, and names shown
    private void dispayPeople(ArrayList<Person> people, Mat frame);
}

public class Person {
    public Rect body;
    public Rect face;
    public String name;

    public Person() {
        name = "Unknown";
    }

    public Person(Rect body) {
        name = "Unknown";
        this.body = body;
    }

    public boolean bodyMatch(Rect newBody) {
        //Check to see if newBody is close enough, so the same person
        //Also update this person's body to the new one if matched
        body = newBody;
    }

    public boolean gotFace() {
        //False if face is null
        //True if overlap between face and body (easy Rect function)
        //If false set face to null, name to Unknown
    }

    public boolean gotName() {
        return !name.equals("Unknown");
    }
}