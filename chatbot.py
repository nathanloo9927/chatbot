from collections import defaultdict
import random
import re
import nltk
from nltk.corpus import wordnet as wn

dst = defaultdict(list)

# update_dst(input): Updates the dialogue state tracker
# Input: A list ([]) of (slot, value) pairs.  Slots should be strings; values can be whatever is
#        most appropriate for the corresponding slot.  Defaults to an empty list.
# Returns: Nothing
def update_dst(input=[]):
    questions = ["symptoms", "family_history", "outside_contact", "other_issues", "no"]
    global dst
    for i, j in input:
        # if there's a "user_intent_history" or "dialogue_state_history" slot in the input, add to the respective slot in dst
        if i == "user_intent_history" or i == "dialogue_state_history":
            dst[i].append(j)
        # if i is one of the prebooking questions, it's only valid answers are "yes" or "no"
        elif i in questions:
            if j == "yes" or j == "no":
                dst[i] = j
        # otherwise, either add it or if it already exists cause of clarification, update it
        else:
            dst[i] = j
    return

# get_dst(slot): Retrieves the stored value for the specified slot, or the full dialogue state at the
#                current time if no argument is provided.
# Input: A string value corresponding to a slot name.
# Returns: A dictionary representation of the full dialogue state (if no slot name is provided), or the
#          value corresponding to the specified slot.
def get_dst(slot=""):
    # if no argument was given (or a blank one was given)
    if slot == "":
        return dict(dst)
    # this is for if the slot doesn't exist with the given input
    elif len(dst[slot]) <= 0:
        return "slot doesn't exist"
    # return the value(s) in the given slot
    else:
        return dst[slot]


# dialogue_policy(dst): Selects the next dialogue state to be uttered by the chatbot.
# Input: A dictionary representation of a full dialogue state.
# Returns: A string value corresponding to a dialogue state, and a list of (slot, value) pairs necessary
#          for generating an utterance for that dialogue state (or an empty list of no (slot, value) pairs
#          are needed).
def dialogue_policy(dst):
    # if the user gave an empty dictionary (or one without a user_intent_history), we give a greeting
    #try:
    unknowns = ["unknown_not_yes_no", "unknown_question", "unknown_time", "unknown_day", "unknown_generic", "unknown_not_specific"]
    if len(dst) <= 0:
        return "greetings", []
    elif "early_exit" in dst["user_intent_history"]:
        return "early_exit", []
    elif dst["user_intent_history"][-1] in unknowns:
        # if we end up here, most likely we're getting the user to repeat his/her response
        return dst["user_intent_history"][-1], []
    elif "symptoms" not in dst:
        return "symptoms", []
    # we should clarify if the user said "yes" or we can move on to the next question on "no"
    elif dst["symptoms"] == "yes" and "clarify_symptoms" not in dst:
        return "clarify_symptoms", []
    # these next ones follow the same pattern as above to ask a question, and if the user answers "yes" to clarify
    elif "family_history" not in dst:
        return "family_history", []
    elif dst["family_history"] == "yes" and "clarify_family_history" not in dst:
        return "clarify_family_history", []
    elif "outside_contact" not in dst:
        return "outside_contact", []
    elif dst["outside_contact"] == "yes" and "clarify_outside_contact" not in dst:
        return "clarify_outside_contact", []
    elif "other_issues" not in dst:
        return "other_issues", []
    elif dst["other_issues"] == "yes" and "clarify_other_issues" not in dst:
        return "clarify_other_issues", []
    # end of prebooking questions, onto actual booking of the apppointment
    # this also loops back to here if the user said no confirming this appointment
    elif "date_and_time" not in dst:
        # reset confirm slot to an empty string in case the user said "no" to confirming the appointment
        update_dst([("confirm", "")])
        return "create_appointment", []
    elif dst["confirm"] == "no":
        update_dst([("confirm", "")])
        return "create_appointment_again", []
    # confirm the given date and time
    # since we set dst["confirm"] = [] in "date_and_time", we know it exists,
    # but it's to check if the user gave an answer to the most recent time given
    elif dst["confirm"] == "":
        return "confirm", [("date_and_time", dst["date_and_time"])]
    # end, book the appointment with the stated time
    elif dst["confirm"] == "yes":
        toreturn = [("date_and_time", dst["date_and_time"])]
        if "clarify_symptoms" in dst["dialogue_state_history"]:
            toreturn.append(("clarify_symptoms", dst["clarify_symptoms"]))
        if "clarify_family_history" in dst["dialogue_state_history"]:
            toreturn.append(("clarify_family_history", dst["clarify_family_history"]))
        if "clarify_outside_contact" in dst["dialogue_state_history"]:
            toreturn.append(("clarify_outside_contact", dst["clarify_outside_contact"]))
        if "clarify_other_issues" in dst["dialogue_state_history"]:
            toreturn.append(("clarify_other_issues", dst["clarify_other_issues"]))
        return "book_appointment", toreturn
    else:
        # if we end up here, we should set it to just terminate, something wrong happened
        return "unknown_question", []


# nlg(state, slots=[]): Generates a surface realization for the specified dialogue act.
# Input: A string indicating a valid state, and optionally a list of (slot, value) tuples.
# Returns: A string representing a sentence generated for the specified state, optionally
#          including the specified slot values if they are needed by the template.
def nlg(state, slots=[]):
    templates = defaultdict(list)

    templates["greetings"].append("Hello, this is Dr. Peng's office. Did you need to schedule an appointment?")
    templates["greetings"].append("Dr. Peng's office. Would you like to book an appointment?")

    templates["unknown_not_yes_no"].append("Sorry, I didn't get that. Could you give me a more definite answer?")
    templates["unknown_not_yes_no"].append("I didn't understand your answer. Could you answer my question more clearly?")

    # not sure if this will be needed, but since I have them in my nlu, I'll include them
    templates["unknown_question"].append("Something went wrong. I'll have to cut it here.")
    templates["unknown_question"].append(
        "Sorry, I will need to end here as something unexpected happened.")

    templates["unknown_time"].append("Your time didn't add up. Could you please repeat the day and time?")
    templates["unknown_time"].append(
        "Your time doesn't make sense. Say your date and time again, but make sure you said it right.")

    templates["unknown_day"].append("I don't understand your timing. Could you give a time I could understand?")
    templates["unknown_day"].append(
        "I didn't catch the date, maybe because it wasn't clear. Could you restate that?")

    templates["unknown_generic"].append("Could you repeat that? I couldn't quite get that")
    templates["unknown_generic"].append("I don't understand. Could you possibly clarify?")

    templates["unknown_not_specific"].append("I don't think what you listed is valid. Could you list them again?")
    templates["unknown_not_specific"].append("I didn't understand what you listed. Could you possibly clarify?")

    # for my dst, I'm not going to add to the dst if we hit one of the unknowns where we ask the user to repeat his/her response
    dontaddtodst = ["unknown_not_yes_no", "unknown_question", "unknown_time", "unknown_day", "unknown_generic", "unknown_not_specific"]

    templates["early_exit"].append("Ok well have a nice day!")
    templates["early_exit"].append("Oh ok well do come back if you actually change your mind")

    templates["symptoms"].append("Ok. Let me ask you a few questions first. Do you have any symptoms of covid?")
    templates["symptoms"].append("Alright. The doctor will need some information first. Do you have any symptoms of covid?")

    templates["clarify_symptoms"].append("Could you describe your symptoms?")
    templates["clarify_symptoms"].append("Could you list them out for me, please?")

    templates["family_history"].append("Any family members you know that have the virus?")
    templates["family_history"].append("Do you know anyone in your family with the virus?")

    templates["clarify_family_history"].append("Who do you know has it?")
    templates["clarify_family_history"].append("Could you list their names and/or relationship to you?")

    templates["outside_contact"].append("Have you been in contact with anyone outside your family who might have the virus?")
    templates["outside_contact"].append("Do you think you came into contact with someone outside who has the virus?")

    templates["clarify_outside_contact"].append("Who did you meet?")
    templates["clarify_outside_contact"].append("Could you describe who it might've been?")

    templates["other_issues"].append("And finally, any other health issues the doctor needs to be aware of?")
    templates["other_issues"].append("Lastly, do you have any other health concerns that should be known?")

    templates["clarify_other_issues"].append("What does the doctor need to know?")
    templates["clarify_other_issues"].append("What should I tell the doctor?")

    templates["create_appointment"].append("Ok. Now what time did you want to book the appointment?")
    templates["create_appointment"].append("Alright that's all the questions I have for now. What time did you want to see the doctor?")

    templates["create_appointment_again"].append("Alright, well then what time did you want to book the appointment?")
    templates["create_appointment_again"].append(
        "Ok, what time did you mean to set the appointment at?")

    templates["confirm"].append("So <date_time>. Is that ok?")
    templates["confirm"].append("I can set you up for <date_time>. That's what you wanted, correct?")

    # Adjustment to the "book appointment" that'll needed to be accounted for in update_dst:
    # originally, it only took the date and time for the (slot, value) input
    # now it takes the clarification statements as well when the user answers yes and elaborates
    # This template will be for if the user gave no symptoms or health issues
    templates["book_appointment"].append("Alright I'll set up the appointment for <date_time> and remind you when the time comes.")
    # These templates will be for if the user gave one of the 2
    templates["book_appointment"].append(
        "Alright I will set an appointment up for <date_time>, and will let the doctor know that you have <symptoms>.")
    templates["book_appointment"].append(
        "Alright I will set an appointment up for <date_time>, and will let the doctor know about your <other_issues>.")
    # This will be for if there's information of both
    templates["book_appointment"].append(
        "Alright I will set an appointment up for <date_time>, and will let the doctor know that you have <symptoms> "
        "as well as <other_issues>.")

    # These templates would be concatenated to one of the above to avoid having to make additional templates for all cases
    # for family history only
    templates["book_appointment"].append(" I also informed the doctor about your <family_member>.")
    # for outside contacts only
    templates["book_appointment"].append(" I also informed the doctor about your contacts with <outside_contact>.")
    # for both
    templates["book_appointment"].append(" I also informed the doctor about your <family_member> and <outside_contact>.")

    i = random.randint(0, 1)
    output = ""
    replacetime = ""
    additionaltemplate = ""
    if len(slots) > 0:
        if len(slots) == 1:
            # the only templates that have only one slot needed to be replaced are ones with date and time
            # the first template in create_appointment is the only template there that only uses one slot
            if state == "book_appointment":
                update_dst([("dialogue_state_history", "book_appointment")])
                output = templates[state][0].replace("<date_time>", str(slots[0][1]))
            else:
                update_dst([("dialogue_state_history", "confirm_appointment")])
                output = templates[state][i].replace("<date_time>", str(slots[0][1]))
        # we're using a book appointment template since it's the only one that uses more than 1 slot
        # so now we have to find out which template we use
        else:
            update_dst([("dialogue_state_history", "book_appointment")])
            needatemplate = True
            templateuse = 0
            basetemplateuse = 0
            templatereplacing = ""
            templatevalue = ""
            basetemplatereplacing = ""
            basetemplatevalue = ""
            if len(slots) == 5: # if we have 5 slots, we know which template we need, otherwise we have to figure out which one to use
                output = templates[state][3]
                needatemplate = False
            for slot, value in slots:
                # same deal as above if we only had one slot needed to be replaced
                if slot == "date_and_time":
                    # since we're selecting the template dependent on how many slots were given,
                    # we should store the value if we don't have a template yet
                    if needatemplate:
                        replacetime = value
                    else:
                        output = output.replace("<date_time>", value)
                # symptoms and issues are used in the core template
                # originally I only had these 2 in the template, but decided to also include the clarification
                # of family history and outside contact as we'll get to below
                elif slot == "clarify_symptoms" or slot == "clarify_other_issues":
                    replacing = ""
                    # if there's only one thing in the list, just put it in the string
                    if len(value) == 1:
                        replacing = value[0]
                    else:
                        for k in value:
                            # if it is the last item in the list, it would be natural to say "and" and not put a comma after the symptom
                            if k == value[-1]:
                                replacing += "and " + k
                            # if there's only 2 items in the list, wouldn't make sense to put a comma
                            elif len(value) == 2:
                                replacing = k + " "
                            # put a comma after the item
                            else:
                                replacing += k + ", "
                    # placeholder for finding which template to use, and update once we find that template
                    if needatemplate:
                        basetemplateuse += 1
                        if basetemplateuse == 2:
                            output = templates[state][3].replace(basetemplatereplacing, basetemplatevalue)
                            # these are information that we will update once we find that template
                            if slot == "clarify_symptoms":
                                output = output.replace("<symptoms>", replacing)
                            elif slot == "clarify_other_issues":
                                output = output.replace("<other_issues>", replacing)
                            needatemplate = False
                        elif basetemplateuse <= 1:
                            if slot == "clarify_symptoms":
                                basetemplatereplacing = "<symptoms>"
                            elif slot == "clarify_other_issues":
                                basetemplatereplacing = "<other_issues>"
                            basetemplatevalue = replacing
                    else:   # a template already exists, build onto that existing template
                        if slot == "clarify_symptoms":
                            output = output.replace("<symptoms>", replacing)
                        elif slot == "clarify_other_issues":
                            output = output.replace("<other_issues>", replacing)
                elif slot == "clarify_family_history" or slot == "clarify_outside_contact":
                    replacing = ""
                    if len(value) == 1:
                        replacing = value[0]
                    else:
                        for k in value:
                            # if it is the last item in the list, it would be natural to say "and" and not put a comma after the symptom
                            if k == value[-1]:
                                replacing += "and " + k
                            # if there's only 2 items in the list, wouldn't make sense to put a comma
                            elif len(value) == 2:
                                replacing = k + " "
                            # put a comma after the item
                            else:
                                replacing += k + ", "
                    # essentially the same as above for symptoms and other issues
                    templateuse += 1
                    if templateuse == 2:
                        additionaltemplate = templates[state][6].replace(templatereplacing, templatevalue)
                        if slot == "clarify_family_history":
                            additionaltemplate = additionaltemplate.replace("<family_member>", replacing)
                        elif slot == "clarify_outside_contact":
                            additionaltemplate = additionaltemplate.replace("<outside_contact>", replacing)
                    elif templateuse <= 1:
                        if slot == "clarify_family_history":
                            templatereplacing = "<family_member>"
                        elif slot == "clarify_outside_contact":
                            templatereplacing = "<outside_contact>"
                        templatevalue = replacing
            # if we need a book appointment template, we grab one based on how many of the clarification slots are used
            if needatemplate:
                if basetemplatereplacing == "<symptoms>":
                    output = templates[state][1].replace(basetemplatereplacing, basetemplatevalue)
                elif basetemplatereplacing == "<other_issues>":
                    output = templates[state][2].replace(basetemplatereplacing, basetemplatevalue)
                output = output.replace("<date_time>", replacetime)
            # build on the main template (as I stated before, this was added after I originally wanted to only include
            # symptoms and other issues in the book appointment statement
            if templateuse == 1:
                if templatereplacing == "<family_member>":
                    additionaltemplate = templates[state][4].replace(templatereplacing, templatevalue)
                elif templatereplacing == "<outside_contact>":
                    additionaltemplate = templates[state][5].replace(templatereplacing, templatevalue)
    else:
        # we can just pick a random template for anything that doesn't have inputs
        output = templates[state][i]
        if state not in dontaddtodst:
            update_dst([("dialogue_state_history", state)])
    # append to base template
    if len(additionaltemplate) > 0:
        output += additionaltemplate

    output = output.replace("<date_time>", replacetime)
    return "Chatbot: " + output


def nlu(input=""):
    slots_and_values = []

    # List of questions where the user responds with a yes or no
    questions = ["symptoms", "family_history", "outside_contact", "other_issues", "confirm_appointment"]

    # Clarification states to make sure the user explains his/her responses
    clarifications = ["clarify_symptoms", "clarify_family_history", "clarify_other_issues", "clarify_outside_contact"]

    user_intent = ""
    # these 4 are specifically for the date and time part
    indexofday = 0
    lastindexofday = 1
    lastindexoftime = 3
    indexoftime = 2

    slotuse = "unknown"
    if "dialogue_state_history" in dst:
        # if we only asked the question, but didn't ask the user to clarify his/her response
        if dst["dialogue_state_history"][-1] in questions:
            # Check to see if the input contains a valid size.
            pattern = re.compile(r"\b([Yy]e.*)|([Ss]ure)|([Yy]up)|([Nn]o.*)|([Nn]a(h)+)|([Oo][Kk].*)|([Dd]on't)\b")
            match = re.search(pattern, input)
            user_intent = "answer_yes_no"
            if match:
                # Find out which question the user was responding to
                if dst["dialogue_state_history"][-1] == "symptoms":
                    slotuse = "symptoms"
                    slots_and_values.append(("user_intent_history", "respond_symptoms"))
                elif dst["dialogue_state_history"][-1] == "family_history":
                    slotuse = "family_history"
                    slots_and_values.append(("user_intent_history", "respond_family_history"))
                elif dst["dialogue_state_history"][-1] == "outside_contact":
                    slotuse = "outside_contact"
                    slots_and_values.append(("user_intent_history", "respond_outside_contact"))
                elif dst["dialogue_state_history"][-1] == "other_issues":
                    slotuse = "other_issues"
                    slots_and_values.append(("user_intent_history", "respond_other_issues"))
                elif dst["dialogue_state_history"][-1] == "confirm_appointment":
                    slotuse = "confirm"
                    slots_and_values.append(("user_intent_history", "confirm"))
                else:
                    user_intent = "unknown"
                    slots_and_values.append(("user_intent_history", "unknown_question"))
            else:
                user_intent = "unknown"
                slots_and_values.append(("user_intent_history", "unknown_not_yes_no"))
        # if the user answered yes to any of those questions (except for confirming the appointment), we wanted
        # the user to go into detail about it
        elif dst["dialogue_state_history"][-1] in clarifications:
            # we only extract certain words from the input that meet the POS criteria based on what question is being answered
            validpos = []
            # for our symptoms and other issues, below is our acceptable POS tags
            # certain words have weird POS tagging. For example, "coughing" is VBG, while "sneezing" is NN
            # we also want to include adjectives and prepositions like "of" in "shortness of breath"
            if dst["dialogue_state_history"][-1] == "clarify_symptoms" or dst["dialogue_state_history"][
                -1] == "clarify_other_issues":
                validpos = ["NN", "NNS", "IN", "JJ", "JJR", "JJS", "VBG"]
                slotuse = dst["dialogue_state_history"][-1]
                slots_and_values.append(("user_intent_history", slotuse))
            # for talking about family history or outside contact, our only acceptable POS tags are names (nouns)
            elif dst["dialogue_state_history"][-1] == "clarify_family_history" or dst["dialogue_state_history"][-1] == "clarify_outside_contact":
                validpos = ["NN", "NNS", "NNP"]
                slotuse = dst["dialogue_state_history"][-1]
                slots_and_values.append(("user_intent_history", slotuse))
            # the user should've given a list of answers or something related, so we assume the user
            # separated those answers with commas
            if "," in input:
                # parse the input into a list were each value is the part after the comma
                listedvalues = input.split(",")
                listofvalues = []
                # and then only get the words that match the POS tag based on what's being clarified
                for i in listedvalues:
                    wordsList = nltk.word_tokenize(i)
                    tagged = nltk.pos_tag(wordsList)
                    toadd = ""
                    for t in tagged:
                        if t[1] in validpos:
                            if len(toadd) <= 0:
                                toadd = t[0]
                            else:
                                toadd = toadd + " " + t[0]
                    if len(toadd) > 0:
                        listofvalues.append(toadd)
                if len(listedvalues) > 0:
                    slots_and_values.append((slotuse, listofvalues))
                else:
                    slots_and_values.append(("user_intent_history", "unknown_not_specific"))
            # sometimes a comma isn't needed, for example if the user only listed 1 or 2 items
            else:
                # if the user listed only 2 items, most likely those 2 items are separated by "and"
                if "and" in input:
                    listedvalues = input.split(" and ")
                    listofvalues = []
                    for i in listedvalues:
                        wordsList = nltk.word_tokenize(i)
                        tagged = nltk.pos_tag(wordsList)
                        toadd = ""
                        for t in tagged:
                            if t[1] in validpos:
                                if len(toadd) <= 0:
                                    toadd = t[0]
                                else:
                                    toadd = toadd + " " + t[0]
                        if len(toadd) > 0:
                            listofvalues.append(toadd)
                    if len(listofvalues) > 0:
                        slots_and_values.append((slotuse, listofvalues))
                    else:
                        slots_and_values.append(("user_intent_history", "unknown_not_specific"))
                else:
                    # most likely, the user only listed one thing
                    wordsList = nltk.word_tokenize(input)
                    tagged = nltk.pos_tag(wordsList)
                    toadd = ""
                    for t in tagged:
                        if t[1] in validpos:
                            if len(toadd) <= 0:
                                toadd = t[0]
                            else:
                                toadd = toadd + " " + t[0]
                    if len(toadd) > 0:
                        slots_and_values.append((slotuse, [toadd]))
                    else:
                        slots_and_values.append(("user_intent_history", "unknown_not_specific"))
        # we asked the user to give us a date and time for the appointment, so we assume we're getting a timeframe
        elif dst["dialogue_state_history"][-1] == "create_appointment" or dst["dialogue_state_history"][-1] == "create_appointment_again":
            # make sure the day is correct
            patternday = re.compile(r"\b([Tt]oday)|([Tt]onight)|([Tt]omorrow)|((([Tt]his|[Nn]ext|[Tt]he following)(\s)+)?([Mm]on|[Tt]ues|[Ww]ednes|[Tt]hurs|[Ff]ri|[Ss]atur|[Ss]un)day)\b")
            matchday = re.search(patternday, input)
            if matchday:
                indexofday = matchday.start()
                lastindexofday = matchday.end()
                # then make sure the time is valid
                patterntime = re.compile(r"\b(((1[0-2])|[0-9])(:[0-5][0-9])?(\s)*([Aa][Mm]|[Pp][Mm]|(in the (morning|afternoon|evening)))|([Aa]fter)?[Nn]oon|[Mm]idnight)\b")
                matchtime = re.search(patterntime, input)
                if matchtime:
                    lastindexoftime = matchtime.end()
                    indexoftime = matchtime.start()
                    user_intent = "give_time"
                    slots_and_values.append(("user_intent_history", "respond_date_and_time"))
                else:
                    slots_and_values.append(("user_intent_history", "unknown_time"))
            else:
                slots_and_values.append(("user_intent_history", "unknown_day"))
        elif dst["dialogue_state_history"][-1] == "book_appointment":
            # most likely the user is saying thank you and goodbye or something like that
            slots_and_values.append(("user_intent_history", "goodbye"))
        elif dst["dialogue_state_history"][-1] == "greetings":
            # since the chatbot greeted the user and asked if he/she'd like to book an appointment
            # it's likely that the user is responding to that question
            # (Would be paired with yes/no questions subset, but made this modification in the end)
            patternyes = re.compile(r"\b([Yy]e.*)|([Oo][Kk].*)|([Ss]ure)|([Yy]up)\b")
            patternno = re.compile(r"\b([Nn]o.*)|([Dd]on't)|([Nn]a(h)+)\b")
            matchyes = re.search(patternyes, input)
            matchno = re.search(patternno, input)
            if matchyes and matchno:
                # gave yes and no in the same sentence, so the input isn't understandable
                slots_and_values.append(("user_intent_history", "unknown_not_yes_no"))
            elif matchyes:
                slots_and_values.append(("user_intent_history", "greetings"))
            elif matchno:
                slots_and_values.append(("user_intent_history", "early_exit"))
            else:
                # if there isn't, we're not really sure what the user is trying to do
                slots_and_values.append(("user_intent_history", "unknown_not_yes_no"))
        else:
            # not really sure what the user is doing
            slots_and_values.append(("user_intent_history", "unknown_generic"))
    # If you're maintaining a dialogue state history but there's nothing there yet, this is probably the
    # first input of the conversation!
    else:
        user_intent = "greetings"
        slots_and_values.append(("user_intent_history", "greetings"))

    # Then, based on what type of user intent you think the user had, you can determine which slot values
    # to try to extract.
    if user_intent == "answer_yes_no":
        pattern = re.compile(r"\b([Yy]e.*)|([Oo]k.*)|([Ss]ure)|([Yy]up)\b")
        answered_yes = re.search(pattern, input)

        pattern = re.compile(r"\b([Nn]o.*)|([Nn]a(h)+)|([Dd]on't)\b")
        answered_no = re.search(pattern, input)

        if answered_yes and not answered_no:
            slots_and_values.append((slotuse, "yes"))
        elif answered_no:
            slots_and_values.append((slotuse, "no"))
    elif user_intent == "give_time":
        # we only want the parts that contain the date and time
        # most likely, the user specified the time after the date
        if indexofday < lastindexoftime:
            slots_and_values.append(("date_and_time", input[indexofday:lastindexoftime]))
        else:
            # there are cases where the user specifies the time first before the day
            slots_and_values.append(("date_and_time", input[indexoftime:lastindexofday]))


    return slots_and_values


def main():
    current_state_tracker = get_dst()
    next_state, slot_values = dialogue_policy(current_state_tracker)
    output = nlg(next_state, slot_values)
    print(output)

    while next_state != "book_appointment" and next_state != "early_exit" and next_state != "unknown_question":
        # Accept the user's input.
        user_input = input("You: ")

        # Perform natural language understanding on the user's input.
        slots_and_values = nlu(user_input)

        # Store the extracted slots and values in the dialogue state tracker.
        update_dst(slots_and_values)

        # Get the full contents of the dialogue state tracker at this time.
        current_state_tracker = get_dst()

        # Determine which state the chatbot should enter next.
        next_state, slot_values = dialogue_policy(current_state_tracker)

        # Generate a natural language realization for the specified state and slot values.
        output = nlg(next_state, slot_values)

        # Print the output to the terminal.
        print(output)

################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
